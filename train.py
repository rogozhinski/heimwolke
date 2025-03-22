import torch
import torchvision
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import cv2
import time
import onnx
import onnxruntime


class WolkeDataset(Dataset):

    def __init__(self, anno_path: str | Path):
        self.anno: np.ndarray = np.loadtxt(anno_path, dtype=str, delimiter=',')
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.input_shape = (224, 224)
        self.anno[:, 1] = LabelEncoder().fit_transform(self.anno[:, 1])

    def __len__(self):
        return self.anno.shape[0]

    def preprocess_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_shape, interpolation=cv2.INTER_NEAREST)
        image = image / 255
        image = (image - self.mean) / self.std
        image = image.transpose((2, 0, 1))
        return image

    def __getitem__(self, item) -> tuple[torch.Tensor, int]:
        image = self.preprocess_image(self.anno[item, 0])
        return torch.tensor(image).to(torch.float32), int(self.anno[item, 1])

class Pipeline:

    def __init__(self, onnx_path: str | Path, learning_rate: float = 0.001):
        self.onnx_path = onnx_path
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = self.get_custom_model()
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def get_custom_model(self, output_size: int = 11):
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(in_features=512, out_features=output_size, bias=True)
        return model

    def validate(self, dataloader: DataLoader, batch_num: int | None = None):
        valid_epoch_loss = 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.loss_function(output, y)
                valid_epoch_loss = (valid_epoch_loss * batch + loss.item()) / (batch + 1)

                if batch_num is not None:
                    if batch == batch_num - 1:
                        break

        return valid_epoch_loss

    def train(
            self,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            epoch_num: int,
            epoch_num_early_stopping: int = 3,
            batch_num: int | None = None,
    ):

        epoch_array = np.empty([0])
        train_loss_array = np.empty([0])
        valid_loss_array = np.empty([0])

        epoch = 1

        min_epoch_loss = np.inf
        epoch_num_without_improvement = 0

        cur_epoch_time = time.time()

        # training by epochs
        while epoch <= epoch_num and epoch_num_without_improvement <= epoch_num_early_stopping:

            train_epoch_loss = 0

            for batch, (x, y) in enumerate(train_dataloader):
                x, y = x.to(self.device), y.to(self.device).to(torch.long)
                output = self.model(x)
                loss = self.loss_function(output, y)
                train_epoch_loss = (train_epoch_loss * batch + loss.item()) / (batch + 1)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_num is not None:
                    if batch == batch_num - 1:
                        break

            valid_epoch_loss = self.validate(val_dataloader, batch_num)
            epoch_array = np.append(epoch_array, epoch)
            train_loss_array = np.append(train_loss_array, train_epoch_loss)
            valid_loss_array = np.append(valid_loss_array, valid_epoch_loss)

            # prints losses every x epochs
            print('epoch ', epoch,
                  '   |   train_epoch_loss:', train_epoch_loss,
                  # '   |   valid_epoch_loss:', valid_epoch_loss,
                  '   |   epochs took', np.round(time.time() - cur_epoch_time), ' s')
            cur_epoch_time = time.time()

            # increases num of epochs without decrease of validation loss
            if valid_epoch_loss > min_epoch_loss:
                epoch_num_without_improvement += 1
            else:
                min_epoch_loss = valid_epoch_loss
                epoch_num_without_improvement = 0

            epoch += 1

        torch.onnx.export(
            model=self.model,
            args=train_dataloader.dataset[0][0][None, ...],
            f=self.onnx_path,
            input_names=['input'],
            export_params=True,
        )

    def onnx_predict(self, input_array: np.ndarray):
        ort_session = onnxruntime.InferenceSession(self.onnx_path)
        outputs = ort_session.run(None, {'input': input_array})
        return outputs


if __name__ == '__main__':

    train_dataset = WolkeDataset(Path(__file__).parent.parent / 'CCSN_v2' / 'train.csv')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4)

    val_dataset = WolkeDataset(Path(__file__).parent.parent / 'CCSN_v2' / 'test.csv')
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=4)

    pipeline = Pipeline(onnx_path=str(Path(__file__).parent / 'models' / 'model.onnx'))
    pipeline.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epoch_num=2,
        batch_num=2,
    )

    print(pipeline.onnx_predict(input_array=val_dataset[0][0].numpy()[None, ...]))
