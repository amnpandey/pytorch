from typing import NamedTuple

import kfp
import kfp.components as components
import kfp.dsl as dsl
from kubernetes import client as k8s_client

def download_dataset(data_dir: str):
    """Download the Fraud Detection data set to the KFP volume to share it among all steps"""
    import subprocess
    import sys
    import time

    subprocess.check_call([sys.executable, "-m", "pip", "install", "minio"])
    time.sleep(5)
    
    import os
    from minio import Minio
    url="minio-acme-iaf.apps.sat.cp.fyre.ibm.com"
    key='minio'
    secret='minio123'

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    print("Directory created successfully")    

    client = Minio(url, key, secret, secure=False)
    client.fget_object('iaf-ai', 'datasets/fraud-detection/dataset.csv', data_dir+'/dataset.csv')
    print("Dataset downloaded successfully.")
    print(os.listdir(data_dir))

def train_model(data_dir: str, model_dir: str):
    """Network class 2-hidden layer model using a pre-downloaded dataset.
    Once trained, the model is persisted to `model_dir`."""

    import subprocess
    import sys
    import time
    import numpy as np
    import pandas as pd
    import os 
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn import metrics
    from pathlib import Path

    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
    time.sleep(5)

    import torch
    print(torch.__version__)
     
    data = pd.read_csv(data_dir+"/dataset.csv")
    
    print(data.head())
    print(data.shape)
    
    x = data.drop('Class', axis=1).values
    y = data['Class'].values
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)   

    bs = 100

    #creating torch dataset and loader using original dataset. 
    #to use resampled dataset, replace ex. xtrain with xtrain_over etc.
    train_ds = torch.utils.data.TensorDataset(torch.tensor(xtrain).float(), torch.tensor(ytrain).float())
    valid_ds = torch.utils.data.TensorDataset(torch.tensor(xtest).float(), torch.tensor(ytest).float())

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=bs)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=bs)

    class Classifier(torch.nn.Module):
        def __init__(self, n_input=10, n_hidden = 20, n_output = 1,drop_prob=0.5):
            super().__init__()
            self.extractor1 = torch.nn.Linear(n_input, n_hidden)
            self.extractor2 = torch.nn.Linear(n_hidden, n_hidden)
            self.relu = torch.nn.ReLU()
            self.drop_out = torch.nn.Dropout(drop_prob)
            self.classifier = torch.nn.Linear(n_hidden, n_output)

        def forward(self, xb):
            x = self.relu(self.extractor1(xb))
            x = self.relu(self.extractor2(x))
            x = self.drop_out(x)
            return self.classifier(x).squeeze()

    def loss_batch(model, loss_func, xb, yb, opt=None):
        loss = loss_func(model(xb), yb)

        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.item(), len(xb)

    #training the network
    def train(epochs, model, loss_func, opt, train_dl, valid_dl):
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_dl:
                loss_batch(model, loss_func, xb, yb, opt)

            model.eval()
            with torch.no_grad():
                losses, nums = zip(
                    *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
                )
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            print(epoch, val_loss) 

    #network setting
    n_input = xtrain.shape[1]
    n_output = 1
    n_hidden = 15

    model = Classifier(n_input=n_input,n_hidden=n_hidden,n_output=n_output,drop_prob=0.2)

    #learning rate
    lr = 0.001

    pos_weight = torch.tensor([5])
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n_epoch = 20

    train(n_epoch,model,loss_func,opt,train_dl,valid_dl)
    model.eval()               
    
    ypred = model(torch.tensor(xtest).float()).detach().numpy()

    ypred [ypred>=0.5] =1.0
    ypred [ypred<0.5] =0.0
    print('Confusion matrix: {}'. format(metrics.confusion_matrix(ytest, ypred)))
    print('AUPRC score: {}'. format(metrics.average_precision_score(ytest, ypred)))
    print('AUROC score: {}'.format(metrics.roc_auc_score(ytest, ypred)))
    print('Accuracy score: {}'.format(metrics.accuracy_score(ytest, ypred)))
    print(metrics.classification_report(ytest, ypred))

    # Create directories if not exists
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir+"/fraud-detection.pt")
    #with open(metrics_path+"/mlpipeline_metrics.json", "w") as f:
    #    json.dump(metrics, f)

def export_model(
    model_dir: str,
    export_bucket: str,
    model_name: str,
    model_version: int,
):
    import os
    import boto3
    from botocore.client import Config

    s3 = boto3.client(
        "s3",
        endpoint_url="http://minio-acme-iaf.apps.sat.cp.fyre.ibm.com",
        aws_access_key_id="minio",
        aws_secret_access_key="minio123",
        config=Config(signature_version="s3v4"),
    )

    # Create export bucket if it does not yet exist
    response = s3.list_buckets()
    export_bucket_exists = False

    for bucket in response["Buckets"]:
        if bucket["Name"] == export_bucket:
            export_bucket_exists = True

    if not export_bucket_exists:
        s3.create_bucket(ACL="public-read-write", Bucket=export_bucket)

    # Save model files to S3
    for root, dirs, files in os.walk(model_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            s3_path = os.path.relpath(local_path, model_dir)

            s3.upload_file(
                local_path,
                export_bucket,
                f"models/{model_name}/{model_version}/{s3_path}",
                ExtraArgs={"ACL": "public-read"},
            )

    response = s3.list_objects(Bucket=export_bucket)
    print(f"All objects in {export_bucket}:")
    for file in response["Contents"]:
        print("{}/{}".format(export_bucket, file["Key"]))

@dsl.pipeline(
    name="Fraud detection Pipeline",
    description="A sample pipeline to demonstrate multi-step model training, evaluation, export",
)
def pipeline(
    url: str = '',
    token: str = '',
    data_dir: str = "/train/data",
    model_dir: str = "/train/model",
    export_bucket: str =  "iaf-ai",
    model_name: str = "fraud-detection",
    model_version: int = 1, 
    metrics_path: str = "/train/metrics"
):
    # create persistent volume
    vop = dsl.VolumeOp(
        name="create-pvc",
        resource_name="fraud-detection-pvc",
        storage_class='csi-cephfs',
        #storage_class='ibmc-file-gold',
        modes=dsl.VOLUME_MODE_RWM,
        size="10Gi"
    )
    # For GPU support, please add the "-gpu" suffix to the base image
    BASE_IMAGE = "mesosphere/kubeflow:1.0.1-0.5.0-tensorflow-2.2.0"

    downloadOp = components.func_to_container_op(
        download_dataset, base_image=BASE_IMAGE
    )(data_dir).add_pvolumes({"/train": vop.volume})

    trainOp = components.func_to_container_op(
        train_model, base_image=BASE_IMAGE
        )(data_dir, model_dir).add_pvolumes({"/train": vop.volume})

    exportOp = components.func_to_container_op(
        export_model, base_image=BASE_IMAGE
        )(model_dir,  export_bucket, model_name, model_version).add_pvolumes({"/train": vop.volume})
    
    trainOp.after(downloadOp)
    exportOp.after(trainOp)

if __name__ == '__main__':      
    from kfp_tekton.compiler import TektonCompiler
    TektonCompiler().compile(pipeline, __file__.replace('.py', '.yaml'))