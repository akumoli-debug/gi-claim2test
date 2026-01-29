import sys


def check_package(name, import_name=None):
    label = import_name or name
    try:
        __import__(import_name or name)
        print(f"✓ {label} installed")
    except ImportError:
        print(f"✗ {label} missing")


def main() -> None:
    print(f"Python version: {sys.version}")
    print()

    check_package("gymnasium")
    check_package("minigrid")
    check_package("stable-baselines3", "stable_baselines3")
    check_package("torch")


if __name__ == "__main__":
    main()

