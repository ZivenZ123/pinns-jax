"""PINN数据模块, 用于管理和组织物理信息神经网络的数据集。"""


class PINNDataModule:
    """物理信息神经网络(PINN)的数据模块。

    这个类负责管理和组织PINN模型所需的所有数据集, 包括训练、验证、测试和预测数据集。
    它实现了PyTorch Lightning的数据模块接口, 提供了标准化的数据加载和处理流程。

    主要功能:
        - 管理多个训练数据集, 支持不同的损失函数
        - 处理验证、测试和预测数据集
        - 支持批处理
        - 为离散网格采样器设置适当的模式

    示例:
        >>> train_dataset = YourDataset()
        >>> val_dataset = YourDataset()
        >>> datamodule = PINNDataModule(train_dataset, val_dataset, batch_size=32)
        >>> datamodule.setup('fit')
    """

    def __init__(
        self,
        train_datasets,
        val_dataset,
        test_dataset=None,
        pred_dataset=None,
        batch_size=None,
    ):
        """初始化一个`PINNDataModule`。

        参数:
            train_datasets: 训练数据集
            val_dataset: 验证数据集

        属性:
            train_datasets: 训练数据集列表
            val_datasets: 验证数据集
            test_datasets: 测试数据集(可选)
            pred_datasets: 预测数据集(可选)
            batch_size: 批处理大小(可选)
            function_mapping: 损失函数映射字典
        """

        self.train_datasets = train_datasets
        self.val_datasets = val_dataset
        self.test_datasets = test_dataset
        self.pred_datasets = pred_dataset

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.pred_data = None
        self.batch_size = batch_size

        self.function_mapping = {}

    def setup(self, stage: str):
        """加载数据。设置变量：`self.data_train`、`self.data_val`、`self.data_test`
        此方法在Lightning调用`trainer.fit()`、`trainer.validate()`、`trainer.test()`和
        `trainer.predict()`之前被调用, 所以要小心不要执行随机分割两次! 此外, 它在
        `self.prepare_data()`之后被调用, 中间有一个屏障, 确保所有进程在数据准备好并
        可供使用后再进入`self.setup()`

        参数:
            stage: 要设置的阶段
                可以是`"fit"`、`"validate"`、`"test"`或`"predict"`。默认为``None``
        """
        if stage == 'fit':
            if not self.train_data:
                self.train_data = {}
                self.set_mode_for_discrete_mesh()

                for train_dataset in self.train_datasets:
                    if self.batch_size is not None:
                        self.train_data[str(train_dataset.loss_fn)] = (
                            train_dataset
                        )
                    else:
                        self.train_data[str(train_dataset.loss_fn)] = (
                            train_dataset[:]
                        )
                    self.function_mapping[str(train_dataset.loss_fn)] = (
                        train_dataset.loss_fn
                    )

        if stage == 'val':
            if not self.val_data:
                self.val_solution_names = []
                self.val_data = {}
                if not isinstance(self.val_datasets, list):
                    self.val_datasets = [self.val_datasets]
                for val_dataset in self.val_datasets:
                    self.val_data[str(val_dataset.loss_fn)] = val_dataset[:]
                    self.function_mapping[str(val_dataset.loss_fn)] = (
                        val_dataset.loss_fn
                    )
                    self.val_solution_names.extend(val_dataset.solution_names)

        if stage == 'test':
            if not self.test_data:
                self.test_data = {}
                if not isinstance(self.test_datasets, list):
                    self.test_datasets = [self.test_datasets]
                for test_dataset in self.test_datasets:
                    self.test_data[str(test_dataset.loss_fn)] = test_dataset[:]
                    self.function_mapping[str(test_dataset.loss_fn)] = (
                        test_dataset.loss_fn
                    )

        if stage == 'pred':
            if not self.pred_data:
                self.pred_data = {}
                if not isinstance(self.pred_datasets, list):
                    self.pred_datasets = [self.pred_datasets]
                for pred_dataset in self.pred_datasets:
                    self.pred_data[str(pred_dataset.loss_fn)] = pred_dataset[:]
                    self.function_mapping[str(pred_dataset.loss_fn)] = (
                        pred_dataset.loss_fn
                    )

    def set_mode_for_discrete_mesh(self):
        """此函数将确定哪些训练数据集用于离散计算。
        然后设置将用于龙格-库塔方法的模式值。
        """

        mesh_idx = [
            (train_dataset.idx_t, train_dataset)
            for train_dataset in self.train_datasets
            if type(train_dataset).__name__ == "DiscreteMeshSampler"
        ]

        mesh_idx = sorted(mesh_idx, key=lambda x: x[0])

        if len(mesh_idx) == 1:
            mesh_idx[0][1].mode = "forward_discrete"
        elif len(mesh_idx) == 2:
            mesh_idx[0][1].mode = "inverse_discrete_1"
            mesh_idx[1][1].mode = "inverse_discrete_2"

        mesh_idx.clear()

    def train_dataloader(self):
        """创建并返回训练数据加载器。

        返回:
            训练数据加载器
        """
        return self.train_data

    def val_dataloader(self):
        """创建并返回验证数据加载器。

        返回:
            验证数据加载器。
        """
        return self.val_data

    def test_dataloader(self):
        """创建并返回测试数据加载器。

        返回:
            测试数据加载器。
        """
        return self.test_data

    def predict_dataloader(self):
        """创建并返回预测数据加载器。

        返回:
            预测数据加载器。
        """
        return self.pred_data


if __name__ == "__main__":
    _ = PINNDataModule(None, None)
