import os
from dotenv import load_dotenv
from typing import Any, Optional, Dict
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig

load_dotenv()
load_dotenv(".local.env", override=True)

class ModelConfiguration(BaseModel):
    """Agent使用模型的配置类。"""
    model_name: Optional[str] = Field(default=None, description="代理使用的模型名称。")
    api_base: Optional[str] = Field(default=None, description="可选的 API 地址。")
    api_key: Optional[str] = Field(default=None, description="可选的 API 密钥。")
    temperature: float = Field(default=0.4, description="模型的温度参数。")
    max_tokens: int = Field(default=8192, description="模型生成的最大 token 数。")

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "ModelConfiguration":
        """从 RunnableConfig 创建一个 ModelConfiguration 实例。"""
        configurable: dict[str, Any] = config.get("configurable", {}) if config else {}

        raw_values: Dict[str, Any] = {
            name: configurable.get(name, field.default)
            for name, field in cls.__fields__.items()
        }

        values = {k: v for k, v in raw_values.items() if v is not None}

        model_name = values["model_name"]
        api_configs = configurable.get("api_configs", {})
        model_api_config = api_configs.get(model_name, api_configs.get("default", {}))

        if "api_base" not in values and "api_base" in model_api_config:
            values["api_base"] = os.getenv(model_api_config["api_base"])
        if "api_key" not in values and "api_key_env" in model_api_config:
            values["api_key"] = os.getenv(model_api_config["api_key_env"])

        return cls(**values)

