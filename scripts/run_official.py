"""Start the official baseline train.py with our dev split (see raynet.official_wrap).

Run: python scripts/run_official.py --official <clone> -- <official train.py arguments>
"""

from raynet.official_wrap import main

if __name__ == "__main__":
    main()
