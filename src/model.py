"""
Wrapper class to train contrastive encoder in Self-Supervised setting.
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
"""

import os
import gc
from tqdm import tqdm
import pandas as pd
import torch as th
import itertools
import statistics as stats
from utils.utils import set_seed
from utils.loss_functions import cpc_loss
from utils.model_plot import save_loss_plot
from utils.model_utils import CPC
th.autograd.set_detect_anomaly(True)  # 正向传播时：开启自动求导的异常侦测


class Model:
    """
    Model: Wrapper class for Contrastive Predictive Coding
    Loss function: InfoNCE - https://arxiv.org/pdf/1807.03748.pdf
    """

    def __init__(self, options):
        """
        :param dict options: Configuration dictionary.
        """
        # Get config
        self.options = options
        # Define which device to use: GPU, or CPU
        self.device = options["device"]
        # Create empty lists and dictionary
        self.model_dict, self.summary = {}, {}
        # Set random seed
        set_seed(self.options)
        # Set hyper-parameters
        self.set_params()
        # Set paths for results as well as initializing some arrays to collect data during training
        self.set_paths()
        # ------Network---------
        # Instantiate networks
        print("Building models...")
        # Set cpc model i.e. setting loss, optimizer, and device assignment (GPU, or CPU)
        self.set_cpc()
        # Print out model architecture
        self.get_model_summary()


    def set_cpc(self):
        """Initialize the model, sets up the loss, optimizer, device assignment (GPU, or CPU) etc."""
        # Instantiate the model
        self.cpc= CPC(self.options)
        # Add the model and its name to a list to save, and load in the future
        self.model_dict.update({"cpc": self.cpc})
        # Assign the model to a device
        self.cpc.to(self.device)
        # Reconstruction loss
        self.cpc_loss = cpc_loss
        # Set optimizer - We will use it only for the parameters of 'online network' + 'predictor'
        self.optimizer_cpc = self._adam()
        # Set scheduler (its usage is optional)
        self.set_scheduler()
        # Add items to summary to be used for reporting later
        self.summary.update({"cpc_loss": []})


    def fit(self, data_loader):
        """
        :param IterableDataset data_loader: Pytorch data loader.
        :return: None

        Fits model to the data using contrastive predictive coding.
        """
        # Training dataset
        train_loader = data_loader.train_loader
        # Validation dataset. Note that it uses only one batch of data to check validation loss to save from computation.
        Xval = self.get_validation_batch(data_loader)
        # Set dictionary and lists to record losses
        self.set_loss_containers()
        # Turn on training mode for each model.
        self.set_mode(mode="training")
        # Batch size
        bs = self.options["batch_size"]
        # Compute total number of batches per epoch
        self.total_batches = len(train_loader)
        print(f"Total number of samples / batches in training set: {len(train_loader.dataset)} / {self.total_batches}")
        # Start training
        for epoch in range(self.options["epochs"]):
            # Attach progress bar to data_loader to check it during training. "leave=True" gives a new line per epoch
            self.train_tqdm = tqdm(enumerate(train_loader), total=self.total_batches, leave=True)
            # Go through batches
            for i, Xbatch in self.train_tqdm:
                # Add channel dim to Xbatch, and turn it into a tensor
                Xbatch = self.process_batch(Xbatch)
                # Initialize hidden units for GRU i.e. all zeros with shape (S, batch size, hidden unit size),
                # S=num_layers*num_directions, i.e. 2*1 since we are using 2 layers, and GRU is set as uni-directional
                hidden = th.zeros((2, bs, self.options["conv_dims"][-1]), device=self.device)
                # Forward pass. Note: Outputs are equivalent to q = concat[qi, qj, dim=0], z = concat[zi, zj, dim=0]
                encoder_samples, predictions, hidden = self.cpc(Xbatch, hidden)
                # Compute reconstruction loss
                cpc_loss, accuracy = self.cpc_loss(encoder_samples, predictions, self.options)
                # Get InfoNCE loss for training per batch
                self.bloss.append(cpc_loss.item())
                # Get accuracy for training per batch
                self.acc_b.append(accuracy)
                # Update the parameters of online network as well as predictor
                self.update_model(cpc_loss, self.optimizer_cpc, retain_graph=True)
                # Clean-up for efficient memory use.
                # gc.collect() 是一个用于手动触发的垃圾回收机制的方法，主要用于强制回收那些无法访问的对象所占用的内存。‌
                # 在Python中，gc.collect()命令通过“标记-清除”和“分代回收”算法来处理循环引用和长期存活的对象的内存管理。
                del cpc_loss, accuracy, encoder_samples, predictions, hidden
                gc.collect()
                # Update log message using epoch and batch numbers
                self.update_log(epoch, i)
            # Record training loss per epoch - which is mean of last batch_size losses
            self.eloss.append(stats.mean(self.bloss[-self.total_batches:-1]))
            # Recording training accuracy per epoch
            self.acc_e.append((stats.mean(self.acc_b[-self.total_batches:-1])))
            # Validate every nth epoch. n=1 by default
            _ = self.validate(Xval) if epoch % self.options["nth_epoch"] == 0 else None
        # Save plot of training and validation losses
        save_loss_plot(self.loss, self._plots_path)
        # Convert loss dictionary to a dataframe
        # 构建一个字典，字典元素有接收的列表中的元素(k, pd.Series(v))组成，
        # 即k: pd.Series(v)。
        # pandas.Series是Pandas库中的一个基本数据结构，类似于一维数组，但带有索引标签。
        # 它主要用于处理和分析数据，具有高效的数据操作能力。‌
        # pandas.Series可以通过pd.Series()构造函数创建，该函数接受多个参数，
        # 包括数据（data）、索引（index）、数据类型（dtype）等。
        # 如果不提供这些参数，则会创建一个空的或默认索引的Series对象。
        # 再通过pd.DataFrame根据字典创建dataframe，
        # 这要求字典中的值都是一个列表，且这些值列表的长度相同，
        # 之后在将其转为表格的形式，即键为标签，值列表则表示不同的标签值
        # 从而方便将这些数据记录到csv表格
        loss_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in self.loss.items()]))
        # Save loss dataframe as csv file for later use
        # .to_csv()用于将DataFrame对象保存为CSV文件
        loss_df.to_csv(self._loss_path + "/losses.csv")

    def set_params(self):
        """Sets up parameters needed for training"""
        # Set learning rate
        self.lr = self.options["learning_rate"]

    def set_loss_containers(self):
        # Initialize empty lists to hold training loss per batch,  training loss per epoch and validation loss per epoch
        self.bloss, self.eloss, self.vloss, self.acc_b, self.acc_e = [], [], [], [], []
        # Loss dictionary: --Initials: "t": training, "v": validation -- Suffixes: "_b": batch, "_e": epoch
        self.loss = {"tloss_b": self.bloss, "tloss_e": self.eloss, "vloss_e": self.vloss, "acc_b": self.acc_b, "acc_e": self.acc_e}
        
    def update_log(self, epoch, batch):
        """Updated the log message displayed during training"""
        # For the first epoch, add losses for batches since we still don't have loss for the epoch
        if epoch < 1:
            description = f"Epoch:[{epoch-1}], Batch:[{batch}] loss / acc:{self.bloss[-1]:.4f} / {self.acc_b[-1]:.4f}"
        # For sub-sequent epochs, display only epoch losses.
        else:
            description = f"Epoch:[{epoch-1}] loss/acc:{self.eloss[-1]:.4f}/{self.acc_e[-1]:.4f}, val loss:{self.vloss[-1]:.4f}"
        # Update the displayed message
        self.train_tqdm.set_description(description)

    def set_mode(self, mode="training"):
        """Sets the mode of the model. If mode==training, the model parameters are expected to be updated."""
        # Change the mode of models, depending on whether we are training them, or using them for evaluation.
        if mode == "training":
            self.cpc.train()
        else:
            self.cpc.eval()

    def process_batch(self, Xbatch):
        """Adds channel dimension, and moves the data to the device as tensor"""
        # Add channel dimension to data
        Xbatch = Xbatch.float().float().unsqueeze(1)
        # Convert the batch to tensor and move it to where the model is
        Xbatch = Xbatch.to(self.device)
        # Return batches
        return Xbatch

    def get_validation_batch(self, data_loader):
        """Wrapper to get validation set. In this case, it uses only the first batch to save from computation time"""
        # 从测试数据加载器中加载并处理一批数据后，将其返回
        # Validation dataset
        validation_loader = data_loader.test_loader
        # Use only the first batch of validation set to save from computation
        Xval = next(iter(validation_loader))
        # Concatenate xi, and xj, and turn it into a tensor
        Xval = self.process_batch(Xval)
        # Return
        return Xval

    def validate(self, Xval):
        """Computes validation loss"""
        with th.no_grad():
            # Turn on evaluation mode
            self.set_mode(mode="evaluation")
            # Initialize hidden units for GRU i.e. all zeros with shape (S, batch size, hidden unit size)
            # S=num_layers*num_directions, i.e. 2*1 since we are using 2 layers, and GRU is set as uni-directional
            hidden = th.zeros((2, self.options["batch_size"], self.options["conv_dims"][-1]), device=self.device)
            # Forward pass on the model
            encoder_samples, predictions, hidden = self.cpc(Xval, hidden)
            # Compute reconstruction loss
            cpc_vloss, _ = cpc_loss(encoder_samples, predictions, self.options)
            # Record validation loss
            self.vloss.append(cpc_vloss.item())
            # Turn on training mode
            self.set_mode(mode="training")
            # Clean up to avoid memory issues
            del cpc_vloss, encoder_samples, predictions, Xval, hidden
            gc.collect()

    def save_weights(self):
        """
        :return: None
        Used to save weights of the model.
        """
        for model_name in self.model_dict:
            th.save(self.model_dict[model_name], self._model_path + "/" + model_name + ".pt")
        print("Done with saving models.")

    def load_models(self):
        """
        :return: None
        Used to load weights saved at the end of the training.
        """
        for model_name in self.model_dict:
            model = th.load(self._model_path + "/" + model_name + ".pt")
            setattr(self, model_name, model.eval())
            print(f"--{model_name} is loaded")
        print("Done with loading models.")

    def get_model_summary(self):
        """
        :return: None
        Sanity check to see if the models are constructed correctly.
        """
        # Summary of the model architecture.
        description  = f"{40*'-'}Summarize models:{40*'-'}\n"
        description += f"{34*'='}{self.options['model_mode'].upper().replace('_', ' ')} Model{34*'='}\n"
        description += f"{self.cpc}\n"
        # Print model architecture
        print(description)

    def update_model(self, loss, optimizer, retain_graph=True):
        """
        :param loss: Loss to be used to compute gradients
        :param optimizer: Optimizer to update weights
        :param retain_graph: If True, keeps computation graph
        :return:
        """
        # Reset optimizer
        optimizer.zero_grad()
        # Backward propagation to compute gradients
        loss.backward(retain_graph=retain_graph)
        # Update weights
        optimizer.step()

    def set_scheduler(self):
        # Set scheduler (Its use will be optional)
        # PyTorch 中的一个学习率调度器，它可以在训练过程中按照一定的策略逐渐降低学习率
        # 将创建的优化器（optimizer）作为参数传入，并指定降低学习率的策略。
        # optimizer：一个 PyTorch 优化器对象，如 torch.optim.SGD、torch.optim.Adam 等。
        # step_size：降低学习率的间隔步数，即经过多少个迭代步骤后降低学习率。
        # gamma：学习率降低的倍数，即学习率每次降低后的缩放比例。
        self.scheduler = th.optim.lr_scheduler.StepLR(self.optimizer_cpc, step_size=2, gamma=0.99)

    def set_paths(self):
        """ Sets paths to bse used for saving results at the end of the training"""
        # Top results directory
        self._results_path = self.options["paths"]["resDir"]
        # Directory to save model
        self._model_path = os.path.join(self._results_path, "training", "model")
        # Directory to save plots as png files
        self._plots_path = os.path.join(self._results_path, "training", "plots")
        # Directory to save losses as csv file
        self._loss_path = os.path.join(self._results_path, "training", "loss")

    def _adam(self):
        """Wrapper for setting up Adam optimizer"""
        # Collect params
        params = [self.cpc.parameters()]
        # Return optimizer
        # itertools.chain()将传入的多个可迭代对象（params是一个或多个可迭代对象，可以是列表、元组、字符串、生成器等。）按顺序连接起来，
        # 返回一个可迭代的链对象（按顺序包含了传入的所有可迭代对象的元素）。
        # 在处理多个可迭代对象时非常有用，它可以避免创建临时列表，提高代码的效率和可读性
        return th.optim.Adam(itertools.chain(*params), lr=self.lr, betas=(0.9, 0.999))

    def _tensor(self, data):
        """Wrapper for moving numpy arrays to the device as a tensor"""
        return th.from_numpy(data).to(self.device).float()
