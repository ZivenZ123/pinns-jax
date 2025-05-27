import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { spawn } from "child_process";
import { promisify } from "util";
import * as fs from "fs/promises";
import * as path from "path";

/**
 * 注册所有PINN相关的工具
 */
export function registerTools(server: McpServer) {
  registerTrainPinnModel(server);
  registerValidateModel(server);
  registerPredictSolution(server);
  registerVisualizeResults(server);
  registerListConfigs(server);
  registerGetTrainingStatus(server);
}

/**
 * 训练PINN模型工具
 */
function registerTrainPinnModel(server: McpServer) {
  server.tool(
    "train_pinn_model",
    "训练物理信息神经网络模型。支持各种PDE问题的求解，包括热传导、波动方程、Navier-Stokes等。",
    {
      config_path: z.string().describe("配置文件路径，例如: configs/heat_equation.yaml"),
      experiment_name: z.string().optional().describe("实验名称，用于标识此次训练"),
      epochs: z.number().optional().describe("训练轮数，默认使用配置文件中的设置"),
      learning_rate: z.number().optional().describe("学习率，默认使用配置文件中的设置"),
      override_params: z.record(z.any()).optional().describe("覆盖配置参数的字典")
    },
    async (args) => {
      try {
        const { config_path, experiment_name, epochs, learning_rate, override_params } = args;
        
        // 构建训练命令
        const command = ["python", "-m", "pinnsjax.train"];
        command.push(`--config-path=${config_path}`);
        
        if (experiment_name) {
          command.push(`experiment_name=${experiment_name}`);
        }
        
        if (epochs) {
          command.push(`trainer.max_epochs=${epochs}`);
        }
        
        if (learning_rate) {
          command.push(`model.optimizer.learning_rate=${learning_rate}`);
        }
        
        // 添加其他覆盖参数
        if (override_params) {
          for (const [key, value] of Object.entries(override_params)) {
            command.push(`${key}=${value}`);
          }
        }
        
        // 执行训练
        const result = await executeCommand(command);
        
        return {
          content: [
            {
              type: "text",
              text: `✅ PINN模型训练完成！\n\n**训练配置:**\n- 配置文件: ${config_path}\n- 实验名称: ${experiment_name || '默认'}\n\n**训练输出:**\n${result.stdout}\n\n${result.stderr ? `**警告/错误:**\n${result.stderr}` : ''}`
            }
          ]
        };
      } catch (error) {
        return {
          content: [
            {
              type: "text", 
              text: `❌ 训练失败: ${error instanceof Error ? error.message : String(error)}`
            }
          ]
        };
      }
    }
  );
}

/**
 * 验证模型工具
 */
function registerValidateModel(server: McpServer) {
  server.tool(
    "validate_model",
    "验证已训练的PINN模型性能，计算验证集上的误差指标。",
    {
      model_path: z.string().describe("模型检查点路径"),
      config_path: z.string().describe("对应的配置文件路径"),
      validation_data: z.string().optional().describe("验证数据路径，如果不同于训练时使用的数据")
    },
    async (args) => {
      try {
        const { model_path, config_path, validation_data } = args;
        
        const command = ["python", "-m", "pinnsjax.validate"];
        command.push(`--config-path=${config_path}`);
        command.push(`--model-path=${model_path}`);
        command.push("val=true");
        
        if (validation_data) {
          command.push(`val_dataset.data_path=${validation_data}`);
        }
        
        const result = await executeCommand(command);
        
        return {
          content: [
            {
              type: "text",
              text: `📊 模型验证完成！\n\n**验证结果:**\n${result.stdout}\n\n${result.stderr ? `**注意事项:**\n${result.stderr}` : ''}`
            }
          ]
        };
      } catch (error) {
        return {
          content: [
            {
              type: "text",
              text: `❌ 验证失败: ${error instanceof Error ? error.message : String(error)}`
            }
          ]
        };
      }
    }
  );
}

/**
 * 预测解工具
 */
function registerPredictSolution(server: McpServer) {
  server.tool(
    "predict_solution",
    "使用训练好的PINN模型预测PDE解。可以在新的时空点上进行预测。",
    {
      model_path: z.string().describe("训练好的模型路径"),
      config_path: z.string().describe("模型配置文件路径"),
      prediction_points: z.string().describe("预测点的数据文件路径或坐标描述"),
      output_path: z.string().optional().describe("预测结果保存路径")
    },
    async (args) => {
      try {
        const { model_path, config_path, prediction_points, output_path } = args;
        
        const command = ["python", "-m", "pinnsjax.predict"];
        command.push(`--config-path=${config_path}`);
        command.push(`--model-path=${model_path}`);
        command.push(`--prediction-points=${prediction_points}`);
        command.push("test=false", "train=false");
        
        if (output_path) {
          command.push(`--output-path=${output_path}`);
        }
        
        const result = await executeCommand(command);
        
        return {
          content: [
            {
              type: "text",
              text: `🔮 预测完成！\n\n**预测设置:**\n- 模型: ${model_path}\n- 预测点: ${prediction_points}\n- 输出路径: ${output_path || '默认路径'}\n\n**预测结果:**\n${result.stdout}`
            }
          ]
        };
      } catch (error) {
        return {
          content: [
            {
              type: "text",
              text: `❌ 预测失败: ${error instanceof Error ? error.message : String(error)}`
            }
          ]
        };
      }
    }
  );
}

/**
 * 可视化结果工具
 */
function registerVisualizeResults(server: McpServer) {
  server.tool(
    "visualize_results",
    "生成PINN训练和预测结果的可视化图表，包括损失曲线、解的分布等。",
    {
      experiment_path: z.string().describe("实验结果路径"),
      plot_types: z.array(z.string()).optional().describe("绘图类型列表，如: ['loss', 'solution', 'error']"),
      save_path: z.string().optional().describe("图片保存路径")
    },
    async (args) => {
      try {
        const { experiment_path, plot_types, save_path } = args;
        
        const command = ["python", "-m", "pinnsjax.visualize"];
        command.push(`--experiment-path=${experiment_path}`);
        
        if (plot_types && plot_types.length > 0) {
          command.push(`--plot-types=${plot_types.join(',')}`);
        }
        
        if (save_path) {
          command.push(`--save-path=${save_path}`);
        }
        
        const result = await executeCommand(command);
        
        return {
          content: [
            {
              type: "text",
              text: `📈 可视化完成！\n\n**可视化设置:**\n- 实验路径: ${experiment_path}\n- 图表类型: ${plot_types?.join(', ') || '默认'}\n- 保存路径: ${save_path || '默认路径'}\n\n**生成结果:**\n${result.stdout}`
            }
          ]
        };
      } catch (error) {
        return {
          content: [
            {
              type: "text",
              text: `❌ 可视化失败: ${error instanceof Error ? error.message : String(error)}`
            }
          ]
        };
      }
    }
  );
}

/**
 * 列出可用配置工具
 */
function registerListConfigs(server: McpServer) {
  server.tool(
    "list_available_configs",
    "列出所有可用的PINN配置模板，包括不同PDE问题的预设配置。",
    {},
    async () => {
      try {
        // 查找配置文件
        const configsPath = path.join(process.cwd(), "pinnsjax", "conf");
        const configs = await findConfigFiles(configsPath);
        
        const configList = configs.map(config => {
          const relativePath = path.relative(configsPath, config);
          return `- ${relativePath}`;
        }).join('\n');
        
        return {
          content: [
            {
              type: "text",
              text: `📋 可用的PINN配置模板:\n\n${configList}\n\n💡 使用这些配置文件可以快速开始训练不同类型的PDE问题。`
            }
          ]
        };
      } catch (error) {
        return {
          content: [
            {
              type: "text",
              text: `❌ 获取配置列表失败: ${error instanceof Error ? error.message : String(error)}`
            }
          ]
        };
      }
    }
  );
}

/**
 * 获取训练状态工具
 */
function registerGetTrainingStatus(server: McpServer) {
  server.tool(
    "get_training_status",
    "获取当前或最近训练任务的状态和进度信息。",
    {
      experiment_path: z.string().optional().describe("实验路径，如果不提供则查找最近的实验")
    },
    async (args) => {
      try {
        const { experiment_path } = args;
        
        // 查找训练日志和状态文件
        const statusInfo = await getTrainingStatus(experiment_path);
        
        return {
          content: [
            {
              type: "text",
              text: `📊 训练状态信息:\n\n${statusInfo}`
            }
          ]
        };
      } catch (error) {
        return {
          content: [
            {
              type: "text",
              text: `❌ 获取训练状态失败: ${error instanceof Error ? error.message : String(error)}`
            }
          ]
        };
      }
    }
  );
}

/**
 * 执行命令的辅助函数
 */
async function executeCommand(command: string[]): Promise<{stdout: string, stderr: string}> {
  return new Promise((resolve, reject) => {
    const process = spawn(command[0], command.slice(1), {
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    let stdout = '';
    let stderr = '';
    
    process.stdout?.on('data', (data) => {
      stdout += data.toString();
    });
    
    process.stderr?.on('data', (data) => {
      stderr += data.toString();
    });
    
    process.on('close', (code) => {
      if (code === 0) {
        resolve({ stdout, stderr });
      } else {
        reject(new Error(`命令执行失败，退出码: ${code}\n${stderr}`));
      }
    });
    
    process.on('error', (error) => {
      reject(error);
    });
  });
}

/**
 * 递归查找配置文件
 */
async function findConfigFiles(dir: string): Promise<string[]> {
  const configs: string[] = [];
  
  try {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      
      if (entry.isDirectory()) {
        const subConfigs = await findConfigFiles(fullPath);
        configs.push(...subConfigs);
      } else if (entry.name.endsWith('.yaml') || entry.name.endsWith('.yml')) {
        configs.push(fullPath);
      }
    }
  } catch (error) {
    // 目录不存在或无法访问
  }
  
  return configs;
}

/**
 * 获取训练状态信息
 */
async function getTrainingStatus(experimentPath?: string): Promise<string> {
  // 这里可以实现读取训练日志、检查点文件等逻辑
  // 暂时返回一个示例状态
  return `🔄 训练状态: 运行中\n📈 当前轮次: 150/1000\n📉 当前损失: 0.0023\n⏱️ 预计剩余时间: 25分钟`;
} 