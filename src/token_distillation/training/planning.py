"""A reviewable recipe; no model load, optimizer step or task submission."""
from __future__ import annotations

from ..config import ProjectConfig


def training_plan(config: ProjectConfig) -> dict:
    return {
        "schema": "distillation_training_plan_v1",
        "status": "pilot_pipeline_available",
        "training_started": False,
        "execution_implemented": True,
        "controlled_ablation_implemented": True,
        "ablation_checkpoint_selection": "common_tune_direction_LogLoss_including_step_zero",
        "internal_heads_receive_normalization_statistics": False,
        "execution_enabled": config.values["training"]["execution_enabled"],
        "explicit_commands_required": ["build-dataset", "train", "evaluate"],
        "context_length": config.context_length,
        "horizon": config.horizon,
        "teacher_schema": config.values["teacher"]["schema"],
        "teacher_scope": "window_local_W32_and_confirmed_native_CZSC_plus_causal_ICT_atomic",
        "student": config.values["student"],
        "ict": config.values["ict"],
        "extra_teacher_dims": {"czsc": [16, 0], **({"ict": [2, 6]} if config.values["ict"]["enabled"] else {})},
        "splits": config.values["splits"],
        "candidate_controls": [
            {"name": "A", "kind": "frozen_Kronos_base"},
            {"name": "B", "kind": "AR_direction_LoRA", "teacher_losses": False},
            {"name": "C", "kind": "AR_direction_LoRA_with_CZSC", "teacher_losses": ["W32", "CZSC_native"]},
            {"name": "D", "kind": "AR_direction_LoRA_with_ICT", "teacher_losses": ["ICT_atomic"]},
            {"name": "E", "kind": "AR_direction_LoRA_with_CZSC_and_ICT", "teacher_losses": ["W32", "CZSC_native", "ICT_atomic"]},
        ],
        "same_for_B_to_E": ["samples", "splits", "training_steps", "seed", "LoRA_capacity", "all_internal_heads", "base_KL"],
        "implemented_components": [
            "read_only_source_bridge", "causal_window", "history_only_normalization",
            "window_local_teacher_preview", "LoRA_injection_merge", "student_internal_heads", "masked_losses",
            "frozen_ICT_source_and_causal_atomic_teacher", "native_finished_CZSC_window_teacher", "multi_teacher_internal_heads",
            "identity_bound_JSONL_dataset", "official_local_asset_loader",
            "frozen_tokenizer_causal_prefix_check", "future_AR_and_teacher_joint_losses",
            "explicit_optimizer_loop_and_trainable_only_checkpoint",
            "tune_only_checkpoint_selection", "chronological_direction_token_and_teacher_evaluation",
            "equal_budget_four_arm_multi_seed_ablation", "paired_initialization_and_sample_order_verification",
            "train_only_teacher_constant_baselines", "paired_origin_date_block_uncertainty",
        ],
        "required_before_training": ["explicit_execution_enabled_config", "explicit_dataset_and_train_commands"],
        "not_implemented": [
            "optimizer_resume", "probability_calibration", "walk_forward_qualification",
            "broker_cost_stress_evaluation", "standalone_merged_student_export_acceptance",
        ],
        "promotion": "pending; no automatic promotion",
    }
