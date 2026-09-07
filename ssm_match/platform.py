"""平台路径适配：云脑（c2net）与本地双模式。

云脑：c2net.context.prepare() 提供统一路径；结果写 output_path，结束前 upload_output()。
本地：全部由 CLI 参数/默认相对路径给出。
"""

import os


class PlatformContext:
    def __init__(self, data_root: str, model_dir: str | None, output_dir: str,
                 in_cloudbrain: bool, c2net_ctx=None):
        self.data_root = data_root
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.in_cloudbrain = in_cloudbrain
        self._c2net = c2net_ctx

    def finish(self):
        if self.in_cloudbrain and self._c2net is not None:
            from c2net.context import upload_output
            upload_output()


def resolve_context(data_root: str | None = None, model_dir: str | None = None,
                    output_dir: str | None = None) -> PlatformContext:
    """本地参数优先于 c2net 自动路径。"""
    c2net_ctx, in_cloudbrain = None, False
    try:
        from c2net.context import prepare
        c2net_ctx = prepare()
        in_cloudbrain = True
    except Exception:
        pass

    if in_cloudbrain:
        data_root = data_root or c2net_ctx.dataset_path
        model_dir = model_dir or os.path.join(c2net_ctx.pretrain_model_path,
                                              "vim-tiny-midclstok")
        output_dir = output_dir or c2net_ctx.output_path
    else:
        data_root = data_root or os.path.join("data")
        model_dir = model_dir or os.environ.get(
            "VIM_CKPT", "/home/zhu/work/lecture/vim-tiny-midclstok")
        output_dir = output_dir or os.path.join("output", "run")

    os.makedirs(output_dir, exist_ok=True)
    return PlatformContext(data_root, model_dir, output_dir, in_cloudbrain, c2net_ctx)
