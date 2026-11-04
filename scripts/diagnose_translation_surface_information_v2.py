"""Same model/steps as V1; cancellation-free likelihood increments."""
from scripts.diagnose_translation_surface_information_v1 import main

if __name__=='__main__':
    main('results/translation_surface_information_v2', stable=True)
