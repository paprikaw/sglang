"""Prepare a distinct, untimed-calibration plus CUDA-only profiling harness."""
from pathlib import Path
import shutil

source=Path(__file__).resolve().parent
target=source.parent/'glm47_speculative_profile_20261023'
target.mkdir(exist_ok=False)
shutil.copytree(source/'runtime',target/'runtime')
shutil.copy(source/'prepare_measure.py',target/'prepare_measure.py')
shutil.copy(source/'mtp_activity.py',target/'runtime/mtp_activity.py')
startup=target/'runtime/startup.py'
text=startup.read_text().replace('    from spec_proof import install as install_graph_proof',
    '    from mtp_activity import install as install_activity\n    install_activity()\n    from spec_proof import install as install_graph_proof')
startup.write_text(text)
hooks=target/'runtime/scheduler_hooks.py'
text=hooks.read_text().replace('        from spec_proof import snapshot',
    "        from mtp_activity import arm\n        if command['action']=='cold':arm()\n        from spec_proof import snapshot")
hooks.write_text(text)
prepare=target/'prepare_measure.py'
text=prepare.read_text().replace("default='spec_r1'", "default='profile_r1'")
text=text.replace("id='glm47-a100-cached-decode-mtp-20261023-'", "id='glm47-a100-cached-decode-mtp-profile-20261023-'")
prepare.write_text(text)
print(target)
