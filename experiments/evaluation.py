"""
实验评估模块 - 对应论文实验验证
提供性能指标计算、基线算法对比和结果可视化
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json

from config import config


@dataclass
class ExperimentResult:
    """实验结果数据结构"""
    algorithm_name: str
    avg_delay: float
    total_energy: float
    data_loss_rate: float
    cache_hit_rate: float
    migration_success_rate: float
    task_completion_rate: float
    
    # 系统效率指标
    cpu_utilization: float
    bandwidth_utilization: float
    queue_utilization: float
    
    # 时延分解
    transmission_delay: float
    processing_delay: float
    waiting_delay: float
    
    # 能耗分解
    computation_energy: float
    communication_energy: float
    migration_energy: float


class BaselineAlgorithms:
    """基线算法实现"""
    
    @staticmethod
    def random_allocation(tasks, nodes):
        """随机分配算法"""
        allocations = {}
        for task in tasks:
            if nodes:
                selected_node = np.random.choice(nodes)
                allocations[task.task_id] = selected_node
        return allocations
    
    @staticmethod
    def greedy_allocation(tasks, node_states):
        """贪心分配算法 - 选择负载最低的节点"""
        allocations = {}
        for task in tasks:
            best_node = None
            min_load = float('inf')
            
            for node_id, state in node_states.items():
                if state.load_factor < min_load:
                    min_load = state.load_factor
                    best_node = node_id
            
            if best_node:
                allocations[task.task_id] = best_node
        
        return allocations
    
    @staticmethod
    def round_robin_allocation(tasks, nodes):
        """轮询分配算法"""
        allocations = {}
        node_index = 0
        
        for task in tasks:
            if nodes:
                allocations[task.task_id] = nodes[node_index % len(nodes)]
                node_index += 1
        
        return allocations
    
    @staticmethod
    def load_aware_allocation(tasks, node_states):
        """负载感知分配算法"""
        allocations = {}
        
        for task in tasks:
            # 计算节点权重 (负载越低权重越高)
            node_weights = {}
            for node_id, state in node_states.items():
                weight = max(0.1, 1.0 - state.load_factor)
                node_weights[node_id] = weight
            
            if node_weights:
                # 按权重概率选择
                nodes = list(node_weights.keys())
                weights = list(node_weights.values())
                
                # 归一化权重
                total_weight = sum(weights)
                if total_weight > 0:
                    probabilities = [w / total_weight for w in weights]
                    selected_node = np.random.choice(nodes, p=probabilities)
                    allocations[task.task_id] = selected_node
        
        return allocations


class PerformanceMetrics:
    """性能指标计算器"""
    
    def __init__(self):
        self.metrics_history = []
        self.baseline_results = {}
        self.default_comm_energy_ratio = 0.3
    
    def calculate_system_metrics(self, vehicles, rsus, uavs, cache_managers, 
                               migration_manager, simulation_time,
                               system_simulator: Optional[Any] = None) -> ExperimentResult:
        """计算系统综合性能指标"""
        
        # 1. 任务时延指标
        all_completed_tasks = []
        for vehicle in vehicles:
            all_completed_tasks.extend(vehicle.processed_tasks)
        for rsu in rsus:
            all_completed_tasks.extend(rsu.processed_tasks)
        for uav in uavs:
            all_completed_tasks.extend(uav.processed_tasks)
        
        if all_completed_tasks:
            avg_delay = np.mean([task.total_delay for task in all_completed_tasks])
            transmission_delays = [sum(task.transmission_delays.values()) 
                                 for task in all_completed_tasks]
            processing_delays = [task.processing_delay for task in all_completed_tasks]
            waiting_delays = [task.waiting_delay for task in all_completed_tasks]
            
            avg_transmission_delay = np.mean(transmission_delays)
            avg_processing_delay = np.mean(processing_delays)
            avg_waiting_delay = np.mean(waiting_delays)
        else:
            avg_delay = 0.0
            avg_transmission_delay = 0.0
            avg_processing_delay = 0.0
            avg_waiting_delay = 0.0
        
        # 2. 能耗指标
        total_computation_energy = 0.0
        
        for vehicle in vehicles:
            total_computation_energy += sum(getattr(vehicle, 'energy_consumption_history', []))
        
        for rsu in rsus:
            total_computation_energy += getattr(rsu.state, 'total_energy', 0.0)
        
        for uav in uavs:
            total_computation_energy += getattr(uav.state, 'total_energy', 0.0)
        
        total_communication_energy = self._estimate_communication_energy(
            vehicles, rsus, uavs, system_simulator)
        if total_communication_energy <= 0 and total_computation_energy > 0:
            total_communication_energy = total_computation_energy * self.default_comm_energy_ratio
        
        total_energy = total_computation_energy + total_communication_energy
        
        # 3. 数据丢失率
        all_generated_tasks = []
        all_dropped_tasks = []
        
        for vehicle in vehicles:
            all_generated_tasks.extend(vehicle.generated_tasks)
            all_dropped_tasks.extend(vehicle.dropped_tasks)
        
        for rsu in rsus:
            all_dropped_tasks.extend(rsu.dropped_tasks)
        
        for uav in uavs:
            all_dropped_tasks.extend(uav.dropped_tasks)
        
        data_loss_rate = len(all_dropped_tasks) / max(1, len(all_generated_tasks))
        task_completion_rate = len(all_completed_tasks) / max(1, len(all_generated_tasks))
        
        # 4. 缓存性能指标
        if cache_managers:
            cache_stats = [manager.get_cache_statistics() for manager in cache_managers.values()]
            avg_cache_hit_rate = np.mean([stats['hit_rate'] for stats in cache_stats])
        else:
            avg_cache_hit_rate = 0.0
        
        # 5. 迁移性能指标
        migration_stats = migration_manager.get_migration_statistics()
        migration_success_rate = migration_stats.get('success_rate', 0.0)
        migration_energy = migration_stats.get('total_downtime', 0.0) * 10  # 简化计算
        
        # 6. 资源利用率
        cpu_utilizations = [node.state.cpu_utilization for node in vehicles + rsus + uavs]
        avg_cpu_utilization = np.mean(cpu_utilizations)
        
        bandwidth_utilizations = [node.state.available_bandwidth / config.communication.total_bandwidth 
                                for node in vehicles + rsus + uavs]
        avg_bandwidth_utilization = np.mean(bandwidth_utilizations)
        
        queue_utilizations = [len([q for q in node.queues.values() if not q.is_empty()]) / len(node.queues)
                            for node in vehicles + rsus + uavs]
        avg_queue_utilization = np.mean(queue_utilizations)
        
        return ExperimentResult(
            algorithm_name="MATD3-MIG",
            avg_delay=float(avg_delay),
            total_energy=total_energy,
            data_loss_rate=data_loss_rate,
            cache_hit_rate=float(avg_cache_hit_rate),
            migration_success_rate=migration_success_rate,
            task_completion_rate=task_completion_rate,
            cpu_utilization=float(avg_cpu_utilization),
            bandwidth_utilization=float(avg_bandwidth_utilization),
            queue_utilization=float(avg_queue_utilization),
            transmission_delay=float(avg_transmission_delay),
            processing_delay=float(avg_processing_delay),
            waiting_delay=float(avg_waiting_delay),
            computation_energy=total_computation_energy,
            communication_energy=total_communication_energy,
            migration_energy=migration_energy
        )
    
    def _estimate_communication_energy(self, vehicles, rsus, uavs, simulator: Optional[Any]) -> float:
        """从节点或仿真器中提取通信能耗，若无则返回0"""
        attr_candidates = (
            'communication_energy_history',
            'comm_energy_history',
            'communication_energy',
            'total_communication_energy'
        )
        total = 0.0
        for node in list(vehicles) + list(rsus) + list(uavs):
            total += self._extract_energy_value(node, attr_candidates)
            state = getattr(node, 'state', None)
            if state is not None:
                total += self._extract_energy_value(state, attr_candidates)
        if total > 0:
            return total
        
        if simulator is not None:
            total = self._extract_energy_value(simulator, attr_candidates)
            if total > 0:
                return total
            metrics_history = getattr(simulator, 'metrics_history', None)
            if metrics_history:
                latest = metrics_history[-1]
                if isinstance(latest, dict):
                    candidate = self._normalize_energy_value(
                        latest.get('total_communication_energy', 0.0))
                    if candidate > 0:
                        return candidate
        return 0.0
    
    def _extract_energy_value(self, obj: Any, attr_candidates: Tuple[str, ...]) -> float:
        if obj is None:
            return 0.0
        for attr in attr_candidates:
            if hasattr(obj, attr):
                value = getattr(obj, attr)
                normalized = self._normalize_energy_value(value)
                if normalized > 0:
                    return normalized
        return 0.0
    
    def _normalize_energy_value(self, value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, np.ndarray):
            return float(np.sum(value))
        if isinstance(value, (list, tuple, set)):
            try:
                return float(sum(value))
            except TypeError:
                return 0.0
        if isinstance(value, dict):
            for key in ('total', 'sum', 'total_energy', 'total_communication_energy'):
                if key in value and isinstance(value[key], (int, float)):
                    return float(value[key])
            numeric_values = [v for v in value.values() if isinstance(v, (int, float))]
            if numeric_values:
                return float(sum(numeric_values))
        return 0.0
    
    def run_baseline_comparison(self, system_simulator, num_steps=100) -> Dict[str, ExperimentResult]:
        """运行基线算法对比实验"""
        baseline_methods = {
            'Random': BaselineAlgorithms.random_allocation,
            'Greedy': BaselineAlgorithms.greedy_allocation,
            'Round_Robin': BaselineAlgorithms.round_robin_allocation,
            'Load_Aware': BaselineAlgorithms.load_aware_allocation
        }
        
        results = {}
        
        print("运行基线算法对比实验...")
        
        for method_name, method_func in baseline_methods.items():
            print(f"  运行 {method_name} 算法...")
            
            # 重置系统状态
            system_simulator._setup_scenario()
            
            # 运行仿真
            start_time = time.time()
            
            for step in range(num_steps):
                # 简化的基线算法仿真
                step_stats = self._simulate_baseline_step(
                    system_simulator, method_func, step)
            
            # 计算指标
            result = self.calculate_system_metrics(
                system_simulator.vehicles,
                system_simulator.rsus, 
                system_simulator.uavs,
                system_simulator.cache_managers,
                system_simulator.migration_manager,
                time.time() - start_time,
                system_simulator
            )
            result.algorithm_name = method_name
            results[method_name] = result
            
            print(f"    完成，平均时延: {result.avg_delay:.3f}s, 成功率: {result.task_completion_rate:.1%}")
        
        return results
    
    def _simulate_baseline_step(self, simulator, allocation_method, step):
        """基线算法仿真步骤"""
        # 收集节点信息
        all_positions = {}
        all_states = {}
        
        for vehicle in simulator.vehicles:
            all_positions[vehicle.node_id] = vehicle.state.position
            all_states[vehicle.node_id] = vehicle.state
        
        for rsu in simulator.rsus:
            all_positions[rsu.node_id] = rsu.state.position
            all_states[rsu.node_id] = rsu.state
        
        for uav in simulator.uavs:
            all_positions[uav.node_id] = uav.state.position
            all_states[uav.node_id] = uav.state
        
        # 车辆生成任务
        all_new_tasks = []
        for vehicle in simulator.vehicles:
            new_tasks, _ = vehicle.step(config.network.time_slot_duration)
            all_new_tasks.extend(new_tasks)
        
        # 使用基线算法分配任务
        if all_new_tasks:
            node_list = list(all_states.keys())
            allocations = allocation_method(all_new_tasks, node_list)
            
            # 简单执行分配 (简化处理)
            for task in all_new_tasks:
                if task.task_id in allocations:
                    target_node = allocations[task.task_id]
                    # 模拟处理成功
                    task.is_completed = True
                    task.completion_time = time.time()
        
        return {"processed_tasks": len(all_new_tasks)}
    
    def generate_comparison_plots(self, results: Dict[str, ExperimentResult], save_path: Optional[str] = None):
        """生成对比图表"""
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        algorithms = list(results.keys())
        
        # 1. 主要性能指标对比
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('MATD3-MIG vs 基线算法性能对比', fontsize=16)
        
        # 平均时延
        delays = [results[alg].avg_delay for alg in algorithms]
        axes[0, 0].bar(algorithms, delays, color=['red' if alg == 'MATD3-MIG' else 'blue' for alg in algorithms])
        axes[0, 0].set_title('平均任务时延 (秒)')
        axes[0, 0].set_ylabel('时延 (s)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 总能耗
        energies = [results[alg].total_energy for alg in algorithms]
        axes[0, 1].bar(algorithms, energies, color=['red' if alg == 'MATD3-MIG' else 'blue' for alg in algorithms])
        axes[0, 1].set_title('总能耗 (焦耳)')
        axes[0, 1].set_ylabel('能耗 (J)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 任务完成率
        completion_rates = [results[alg].task_completion_rate for alg in algorithms]
        axes[0, 2].bar(algorithms, completion_rates, color=['red' if alg == 'MATD3-MIG' else 'blue' for alg in algorithms])
        axes[0, 2].set_title('任务完成率')
        axes[0, 2].set_ylabel('完成率')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 数据丢失率
        loss_rates = [results[alg].data_loss_rate for alg in algorithms]
        axes[1, 0].bar(algorithms, loss_rates, color=['red' if alg == 'MATD3-MIG' else 'blue' for alg in algorithms])
        axes[1, 0].set_title('数据丢失率')
        axes[1, 0].set_ylabel('丢失率')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 缓存命中率
        cache_rates = [results[alg].cache_hit_rate for alg in algorithms]
        axes[1, 1].bar(algorithms, cache_rates, color=['red' if alg == 'MATD3-MIG' else 'blue' for alg in algorithms])
        axes[1, 1].set_title('缓存命中率')
        axes[1, 1].set_ylabel('命中率')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # CPU利用率
        cpu_utils = [results[alg].cpu_utilization for alg in algorithms]
        axes[1, 2].bar(algorithms, cpu_utils, color=['red' if alg == 'MATD3-MIG' else 'blue' for alg in algorithms])
        axes[1, 2].set_title('CPU利用率')
        axes[1, 2].set_ylabel('利用率')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 时延分解图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        matd3_result = results.get('MATD3-MIG', results[list(results.keys())[0]])
        
        delay_components = ['传输时延', '处理时延', '等待时延']
        delay_values = [
            matd3_result.transmission_delay,
            matd3_result.processing_delay, 
            matd3_result.waiting_delay
        ]
        
        ax.pie(delay_values, labels=delay_components, autopct='%1.1f%%', startangle=90)
        ax.set_title('MATD3-MIG时延分解')
        
        if save_path:
            plt.savefig(f"{save_path}/delay_breakdown.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. 能耗分解图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        energy_components = ['计算能耗', '通信能耗', '迁移能耗']
        energy_values = [
            matd3_result.computation_energy,
            matd3_result.communication_energy,
            matd3_result.migration_energy
        ]
        
        ax.pie(energy_values, labels=energy_components, autopct='%1.1f%%', startangle=90)
        ax.set_title('MATD3-MIG能耗分解')
        
        if save_path:
            plt.savefig(f"{save_path}/energy_breakdown.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results_to_file(self, results: Dict[str, ExperimentResult], filename: str):
        """保存实验结果到文件"""
        
        # 转换为DataFrame
        data = []
        for alg_name, result in results.items():
            data.append({
                'Algorithm': alg_name,
                'Avg_Delay': result.avg_delay,
                'Total_Energy': result.total_energy,
                'Data_Loss_Rate': result.data_loss_rate,
                'Cache_Hit_Rate': result.cache_hit_rate,
                'Migration_Success_Rate': result.migration_success_rate,
                'Task_Completion_Rate': result.task_completion_rate,
                'CPU_Utilization': result.cpu_utilization,
                'Bandwidth_Utilization': result.bandwidth_utilization,
                'Queue_Utilization': result.queue_utilization
            })
        
        df = pd.DataFrame(data)
        
        # 保存为CSV
        df.to_csv(f"{filename}.csv", index=False)
        
        # 保存为JSON
        results_dict = {}
        for alg_name, result in results.items():
            results_dict[alg_name] = {
                'avg_delay': result.avg_delay,
                'total_energy': result.total_energy,
                'data_loss_rate': result.data_loss_rate,
                'cache_hit_rate': result.cache_hit_rate,
                'migration_success_rate': result.migration_success_rate,
                'task_completion_rate': result.task_completion_rate
            }
        
        with open(f"{filename}.json", 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        print(f"实验结果已保存到 {filename}.csv 和 {filename}.json")
    
    def print_performance_summary(self, results: Dict[str, ExperimentResult]):
        """打印性能摘要"""
        print("\n" + "="*80)
        print("性能对比摘要")
        print("="*80)
        
        print(f"{'算法':<15} {'平均时延(s)':<12} {'总能耗(J)':<12} {'完成率':<10} {'丢失率':<10} {'缓存命中率':<12}")
        print("-" * 80)
        
        for alg_name, result in results.items():
            print(f"{alg_name:<15} {result.avg_delay:<12.3f} {result.total_energy:<12.1f} " +
                  f"{result.task_completion_rate:<10.1%} {result.data_loss_rate:<10.1%} " +
                  f"{result.cache_hit_rate:<12.1%}")
        
        # 计算MATD3-MIG相对于最佳基线的改进
        if 'MATD3-MIG' in results:
            matd3_result = results['MATD3-MIG']
            baseline_results = {k: v for k, v in results.items() if k != 'MATD3-MIG'}
            
            if baseline_results:
                best_delay = min(r.avg_delay for r in baseline_results.values())
                best_energy = min(r.total_energy for r in baseline_results.values())
                best_completion = max(r.task_completion_rate for r in baseline_results.values())
                
                delay_improvement = (best_delay - matd3_result.avg_delay) / best_delay * 100
                energy_improvement = (best_energy - matd3_result.total_energy) / best_energy * 100
                completion_improvement = (matd3_result.task_completion_rate - best_completion) / best_completion * 100
                
                print("\n" + "="*80)
                print("MATD3-MIG 相对改进:")
                print(f"  时延改进: {delay_improvement:+.1f}%")
                print(f"  能耗改进: {energy_improvement:+.1f}%") 
                print(f"  完成率改进: {completion_improvement:+.1f}%")
                print("="*80)


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, system_simulator):
        self.simulator = system_simulator
        self.metrics_calculator = PerformanceMetrics()
    
    def run_full_evaluation(self, num_steps=100, save_results=True):
        """运行完整评估实验"""
        print("开始MATD3-MIG系统完整评估...")
        
        # 1. 运行MATD3-MIG算法
        print("\n1. 运行MATD3-MIG算法...")
        self.simulator.run_complete_simulation(num_steps)
        
        matd3_result = self.metrics_calculator.calculate_system_metrics(
            self.simulator.vehicles,
            self.simulator.rsus,
            self.simulator.uavs, 
            self.simulator.cache_managers,
            self.simulator.migration_manager,
            num_steps * config.network.time_slot_duration,
            self.simulator
        )
        
        # 2. 运行基线算法对比
        print("\n2. 运行基线算法对比...")
        baseline_results = self.metrics_calculator.run_baseline_comparison(
            self.simulator, num_steps)
        
        # 3. 合并所有结果
        all_results = {'MATD3-MIG': matd3_result}
        all_results.update(baseline_results)
        
        # 4. 打印性能摘要
        self.metrics_calculator.print_performance_summary(all_results)
        
        # 5. 生成图表
        print("\n3. 生成对比图表...")
        save_path_str = "results" if save_results else None
        self.metrics_calculator.generate_comparison_plots(all_results, save_path_str)
        
        # 6. 保存结果
        if save_results:
            print("\n4. 保存实验结果...")
            self.metrics_calculator.save_results_to_file(all_results, "results/experiment_results")
        
        print("\n✓ 完整评估实验完成!")
        
        return all_results
