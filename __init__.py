from .sim      import Simulator, HardwareConfig, compare_hardware
from .analyze  import compute_speedup_table, memory_pressure_report, suite_summary
from .visualize import save_all, plot_roofline, plot_runtime_comparison
from .core     import (
    MatMulWorkload, Conv2DWorkload, AttentionWorkload, ElementwiseWorkload,
    llm_inference_suite, vision_model_suite, matmul_scaling_suite,
    cpu_desktop_hierarchy, npu_hierarchy, mobile_hierarchy,
    cpu_avx512, npu_systolic, mobile_arm_neon,
)
