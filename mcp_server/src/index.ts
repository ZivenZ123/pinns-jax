#!/usr/bin/env node

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { registerTools } from "./tools/index.js";
import { registerResources } from "./resources/index.js";
import { registerPrompts } from "./prompts/index.js";

/**
 * PINNs-JAX MCP Server
 * 
 * 这个MCP Server为PINNs-JAX库提供AI助手接口，支持：
 * - 训练物理信息神经网络模型
 * - 模型验证和测试
 * - 结果预测和可视化
 * - 实验管理和配置
 */

// 创建服务器实例
const server = new McpServer(
  {
    name: "pinnsjax-mcp-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
      resources: {},
      prompts: {},
    },
  }
);

// 注册所有功能
registerTools(server);
registerResources(server);
registerPrompts(server);

// 主函数
async function main() {
  // 创建标准输入输出传输
  const transport = new StdioServerTransport();
  
  // 连接服务器和传输
  await server.connect(transport);
  
  console.error("PINNs-JAX MCP Server 已启动");
}

// 启动服务器
main().catch((error) => {
  console.error("服务器启动失败:", error);
  process.exit(1);
}); 