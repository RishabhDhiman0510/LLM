import sys

import uvicorn

from .server import create_app
from .settings import settings


def main() -> None:
    if not (settings.model_dir / "model.pt").exists():
        sys.exit(
            f"Model bundle not found in {settings.model_dir}\n"
            "Download wikigpt-12m.zip from the output of Kaggle notebook 03 and extract it into the models folder."
        )
    print(f"WikiGPT is running at http://{settings.host}:{settings.port}  (Ctrl+C to stop)")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
