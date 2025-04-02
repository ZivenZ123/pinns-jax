"""训练器模块，提供模型训练、验证、测试和预测的功能。"""

import time
from tqdm import tqdm
from pinnsjax import utils

log = utils.get_pylogger(__name__)


class Trainer:
    """训练器类，用于管理模型训练过程"""

    def __init__(self,
                 max_epochs,
                 min_epochs: int = 1,
                 enable_progress_bar: bool = True,
                 check_val_every_n_epoch: int = 1,
                 accelerator: str = 'cpu',
                 default_root_dir: str = "",
                 lbfgs=None):
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.enable_progress_bar = enable_progress_bar
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.callback_metrics = {}
        self.current_epoch = 0
        self.lbfgs = lbfgs
        self.accelerator = accelerator
        self.default_root_dir = default_root_dir
        self.time_list = []

    def callback_pbar(
        self, loss_name, loss, extra_variables=None,
        extra_variables_values=None
    ):
        """更新进度条显示的回调函数"""
        res = f"{loss_name}: {loss:.4f}"
        self.callback_metrics[loss_name] = loss

        if extra_variables:
            disc = []
            for key in extra_variables.keys():
                disc.append(f"{key}: {extra_variables_values[key]:.4f}")
                self.callback_metrics[key] = extra_variables_values[key]

            extra_info = ', '.join(disc)
            res = f"{res}, {extra_info}"

        return res

    def set_callback_metrics(
        self, loss_name, loss, extra_variables=None,
        extra_variables_values=None
    ):
        """设置回调指标"""
        self.callback_metrics[loss_name] = loss

        if extra_variables:
            for key in extra_variables.keys():
                self.callback_metrics[key] = extra_variables_values[key]

    def initalize_tqdm(self, max_epochs):
        """初始化进度条"""
        return tqdm(
            total=max_epochs,
            bar_format=(
                "{n_fmt}/{total_fmt} {percentage:3.0f}%|{bar}|"
                "[已用时间: {elapsed}, 剩余时间: {remaining}, "
                "{rate_fmt}{postfix}, {desc}]"
            ),
        )

    def fit(self, model, datamodule):
        """训练模型的主函数"""
        datamodule.setup('fit')
        datamodule.setup('val')

        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()

        model.val_solution_names = datamodule.val_solution_names
        model.function_mapping = datamodule.function_mapping

        if self.enable_progress_bar:
            self.pbar = self.initalize_tqdm(self.max_epochs)

        log.info("使用Adam优化器进行训练")

        self.current_index = []
        self.dataset_size = []

        if datamodule.batch_size is not None:
            for i, (key, data) in enumerate(train_dataloader.items()):
                self.current_index.append(0)
                self.dataset_size.append(len(data))
                data.shuffle()

        for epoch in range(self.current_epoch, self.max_epochs):
            start_time = time.time()

            if datamodule.batch_size is not None:
                train_data = {}
                for i, (key, data) in enumerate(train_dataloader.items()):
                    batch_end = self.current_index[i] + datamodule.batch_size
                    train_data[key] = data[self.current_index[i]:batch_end]
                    self.current_index[i] = batch_end

                    if (self.current_index[i] + datamodule.batch_size >=
                            self.dataset_size[i]):
                        data.shuffle()
                        self.current_index[i] = 0

                train_step_result = model.train_step(
                    model.trainable_variables,
                    model.opt_state,
                    train_data
                )
                (loss, extra_variables, model.trainable_variables,
                 model.opt_state) = train_step_result

            else:
                train_step_result = model.train_step(
                    model.trainable_variables,
                    model.opt_state,
                    train_dataloader
                )
                (loss, extra_variables, model.trainable_variables,
                 model.opt_state) = train_step_result

            elapsed_time = time.time() - start_time
            self.time_list.append(elapsed_time)

            self.set_callback_metrics(
                'train/loss',
                loss,
                extra_variables,
                model.trainable_variables
            )

            if self.enable_progress_bar:
                self.pbar.update(1)
                description = self.callback_pbar(
                    'train/loss',
                    loss,
                    extra_variables,
                    model.trainable_variables
                )
                self.pbar.set_description(description)
                self.pbar.refresh()

            if epoch % self.check_val_every_n_epoch == 0:
                loss, error_dict = model.validation_step(val_dataloader)
                self.set_callback_metrics('val/loss', loss)

                if self.enable_progress_bar:
                    descriptions = [self.callback_pbar('val/loss', loss)]
                    for error_name in model.val_solution_names:
                        error_value = error_dict[error_name]
                        descriptions.append(
                            self.callback_pbar(
                                f'val/error_{error_name}',
                                error_value
                            )
                        )

                    full_description = ', '.join(descriptions)
                    self.pbar.set_postfix_str(full_description)
                    self.pbar.refresh()

        if self.enable_progress_bar:
            self.pbar.close()

        if self.lbfgs:
            log.info("使用L-BFGS-B优化器进行训练")
            # 注意：这里需要导入lbfgs_minimize函数
            from pinnsjax.optimizers import lbfgs_minimize
            lbfgs_minimize(model, data, self.callback_pbar)

    def validate(self, model, datamodule):
        """验证模型"""
        datamodule.setup('val')
        data = datamodule.val_dataloader()

        loss, error_dict = model.validation_step(data)

        for key, error in error_dict.items():
            self.set_callback_metrics(f'val/error_{key}', error)

        return loss, error_dict

    def predict(self, model, datamodule):
        """模型预测"""
        datamodule.setup('pred')
        data = datamodule.predict_dataloader()

        preds = model.predict_step(data)

        return preds

    def test(self, model, datamodule):
        """模型测试"""
        datamodule.setup('test')
        data = datamodule.test_dataloader()

        log.info("开始测试")

        loss, error_dict = model.test_step(data)

        for key, error in error_dict.items():
            self.set_callback_metrics(f'test/error_{key}', error)

        return loss, error_dict
