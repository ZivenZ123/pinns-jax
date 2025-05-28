"""PINNs-JAX MCP Server

这个包为PINNs-JAX库提供Model Context Protocol (MCP) 服务器功能，
允许AI助手与物理信息神经网络进行交互。
"""

from .server import mcp

__version__ = "0.1.0"
__all__ = ["mcp"]
