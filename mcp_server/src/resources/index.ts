import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import * as fs from "fs/promises";
import * as path from "path";

/**
 * 注册所有PINN相关的资源
 */
export function registerResources(server: McpServer) {
  registerTrainingHistory(server);
  registerModelConfigs(server);
  registerExperimentResults(server);
}

/**
 * 训练历史资源
 */
function registerTrainingHistory(server: McpServer) {
  server.resource(
    "training_history",
    "存储PINN模型的训练历史和指标数据，包括损失曲线、验证误差等。",
    async () => {
      try {
        const historyPath = path.join(process.cwd(), "data", "training_history.json");
        
        let history = {};
        try {
          const data = await fs.readFile(historyPath, "utf-8");
          history = JSON.parse(data);
        } catch {
          // 文件不存在，返回空历史
          history = {
            experiments: [],
            last_updated: new Date().toISOString(),
            total_experiments: 0
          };
        }
        
        return {
          contents: [
            {
              type: "text",
              text: `# PINN训练历史\n\n${JSON.stringify(history, null, 2)}`
            }
          ]
        };
      } catch (error) {
        return {
          contents: [
            {
              type: "text",
              text: `❌ 无法读取训练历史: ${error instanceof Error ? error.message : String(error)}`
            }
          ]
        };
      }
    }
  );
}

/**
 * 模型配置资源
 */
function registerModelConfigs(server: McpServer) {
  server.resource(
    "model_configs",
    "管理PINN模型的配置模板和参数设置，支持不同PDE问题的配置。",
    async () => {
      try {
        const configsPath = path.join(process.cwd(), "pinnsjax", "conf");
        const configs = await loadAllConfigs(configsPath);
        
        return {
          contents: [
            {
              type: "text",
              text: `# PINN模型配置\n\n## 可用配置模板\n\n${configs.map(config => 
                `### ${config.name}\n- 路径: ${config.path}\n- 描述: ${config.description}\n- 参数: ${JSON.stringify(config.params, null, 2)}\n`
              ).join('\n')}`
            }
          ]
        };
      } catch (error) {
        return {
          contents: [
            {
              type: "text",
              text: `❌ 无法读取模型配置: ${error instanceof Error ? error.message : String(error)}`
            }
          ]
        };
      }
    }
  );
}

/**
 * 实验结果资源
 */
function registerExperimentResults(server: McpServer) {
  server.resource(
    "experiment_results",
    "存储PINN实验的结果数据，包括模型性能、预测精度等指标。",
    async () => {
      try {
        const resultsPath = path.join(process.cwd(), "outputs");
        const experiments = await loadExperimentResults(resultsPath);
        
        return {
          contents: [
            {
              type: "text",
              text: `# PINN实验结果\n\n## 最近实验\n\n${experiments.map(exp => 
                `### 实验: ${exp.name}\n- 时间: ${exp.timestamp}\n- 状态: ${exp.status}\n- 最佳损失: ${exp.best_loss}\n- 验证误差: ${exp.validation_error}\n- 模型路径: ${exp.model_path}\n`
              ).join('\n')}`
            }
          ]
        };
      } catch (error) {
        return {
          contents: [
            {
              type: "text",
              text: `❌ 无法读取实验结果: ${error instanceof Error ? error.message : String(error)}`
            }
          ]
        };
      }
    }
  );
}

/**
 * 加载所有配置文件
 */
async function loadAllConfigs(configsPath: string) {
  const configs = [];
  
  try {
    const entries = await fs.readdir(configsPath, { withFileTypes: true });
    
    for (const entry of entries) {
      if (entry.isFile() && (entry.name.endsWith('.yaml') || entry.name.endsWith('.yml'))) {
        const configPath = path.join(configsPath, entry.name);
        try {
          const content = await fs.readFile(configPath, 'utf-8');
          configs.push({
            name: entry.name,
            path: configPath,
            description: extractDescription(content),
            params: extractMainParams(content)
          });
        } catch {
          // 跳过无法读取的文件
        }
      }
    }
  } catch {
    // 目录不存在
  }
  
  return configs;
}

/**
 * 加载实验结果
 */
async function loadExperimentResults(resultsPath: string) {
  const experiments = [];
  
  try {
    const entries = await fs.readdir(resultsPath, { withFileTypes: true });
    
    for (const entry of entries) {
      if (entry.isDirectory()) {
        const expPath = path.join(resultsPath, entry.name);
        try {
          const metricFile = path.join(expPath, "metrics.json");
          const metricsContent = await fs.readFile(metricFile, 'utf-8');
          const metrics = JSON.parse(metricsContent);
          
          experiments.push({
            name: entry.name,
            timestamp: entry.name.split('_')[0] || 'unknown',
            status: 'completed',
            best_loss: metrics.best_loss || 'N/A',
            validation_error: metrics.validation_error || 'N/A',
            model_path: path.join(expPath, "checkpoints")
          });
        } catch {
          // 跳过无法读取的实验
          experiments.push({
            name: entry.name,
            timestamp: 'unknown',
            status: 'incomplete',
            best_loss: 'N/A',
            validation_error: 'N/A',
            model_path: 'N/A'
          });
        }
      }
    }
  } catch {
    // 目录不存在
  }
  
  return experiments.slice(0, 10); // 只返回最近10个实验
}

/**
 * 从配置文件中提取描述
 */
function extractDescription(content: string): string {
  const lines = content.split('\n');
  for (const line of lines) {
    if (line.trim().startsWith('#') && line.includes('描述') || line.includes('description')) {
      return line.replace('#', '').trim();
    }
  }
  return '无描述';
}

/**
 * 从配置文件中提取主要参数
 */
function extractMainParams(content: string): Record<string, any> {
  // 简单的YAML解析，提取主要参数
  const params: Record<string, any> = {};
  const lines = content.split('\n');
  
  for (const line of lines) {
    if (line.includes(':') && !line.trim().startsWith('#')) {
      const [key, value] = line.split(':').map(s => s.trim());
      if (key && value && !key.startsWith('-')) {
        params[key] = value;
      }
    }
  }
  
  return params;
} 