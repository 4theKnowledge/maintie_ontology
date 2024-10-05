import os
from dataclasses import dataclass


@dataclass
class Config:
    # OwlReady2 configuration
    JAVA_PATH: str

    # Processing configuration
    CREATE_FUNCTION_STATES: bool = False
    CONVERT_UNDESIRABLE_STATE_TO_INDETERMINATE_STATE: bool = False


# if os.name == 'nt':  # Windows
java_path = r"C:\Program Files\Java\jre-1.8\bin\java.exe"
# else:  # Linux or other
#     java_path = "/usr/bin/java"

config = Config(JAVA_PATH=java_path)
