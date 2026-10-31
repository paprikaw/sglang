"""Prepare only after all source production/calibrations are complete; no GPU launch."""
import sys
from prepare_measure import main
if __name__=='__main__':
    if '--profile-source-run' not in sys.argv:
        raise SystemExit('--profile-source-run is mandatory; this entry point cannot run production')
    main()
