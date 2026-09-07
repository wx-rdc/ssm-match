from .matching import CrossImageMatcher
from .ssm_match import SCALES, SCALE_SPECS, SSMMatch

__all__ = ["CrossImageMatcher", "SSMMatch", "SCALES", "SCALE_SPECS",
           "build_model_from_config"]


def build_model_from_config(cfg: dict, n_way: int | None = None) -> SSMMatch:
    """从训练配置 dict 构造 SSMMatch（缺省键 = 完整模型）；train/eval 共用。

    因子化消融开关（实验矩阵 B）与各键含义见 configs/ablation_template.yaml 注释。
    """
    return SSMMatch(
        n_way=n_way if n_way is not None else cfg["n_way"],
        query_chunk=cfg.get("query_chunk", 5),
        aux_weight=cfg.get("aux_weight", 0.1),
        d_state=cfg.get("d_state", 16),
        scales=tuple(cfg.get("scales", ("f1", "f2", "f3"))),
        use_prior=cfg.get("use_prior", True),
        use_sep=cfg.get("use_sep", True),
        scan_direction=cfg.get("scan_direction", "bi"),
        use_dir_gate=cfg.get("use_dir_gate", True),
        use_evi_gate=cfg.get("use_evi_gate", True),
        score_sigmoid=cfg.get("score_sigmoid", False),
    )
