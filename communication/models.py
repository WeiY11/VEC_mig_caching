"""
通信与计算模型 - 对应论文第5节
实现VEC系统中的无线通信模型和计算能耗模型
"""
import numpy as np
import math
from typing import Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass

from models.data_structures import Position, CommunicationLink, Task
from config import config
from utils import db_to_linear, linear_to_db, dbm_to_watts


@dataclass
class ChannelState:
    """信道状态信息"""
    distance: float = 0.0
    los_probability: float = 0.0
    path_loss_db: float = 0.0
    shadowing_db: float = 0.0
    channel_gain_linear: float = 0.0
    interference_power: float = 0.0


class WirelessCommunicationModel:
    """
    无线通信模型 - 对应论文第5.2节
    实现3GPP标准的VEC无线通信信道模型
    """
    
    def __init__(self):
        # 3GPP标准通信参数
        self.carrier_frequency = 2.0e9  # 2 GHz - 3GPP标准频率
        self.los_threshold = 50.0  # d_0 = 50m - 3GPP TS 38.901
        self.los_decay_factor = 100.0  # α_LoS = 100m - 3GPP标准
        self.shadowing_std_los = 4.0  # X_σ,LoS = 4 dB - 3GPP标准
        self.shadowing_std_nlos = 8.0  # X_σ,NLoS = 8 dB - 3GPP标准
        self.coding_efficiency = 0.8  # η_coding - 编码效率
        self.processing_delay = 0.001  # T_proc = 1ms - 处理时延
        self.thermal_noise_density = -174.0  # dBm/Hz - 热噪声密度
        
        # 3GPP天线增益参数
        self.antenna_gain_rsu = 15.0  # 15 dBi - RSU天线增益
        self.antenna_gain_uav = 5.0   # 5 dBi - UAV天线增益
        self.antenna_gain_vehicle = 3.0  # 3 dBi - 车辆天线增益
        self.fast_fading_factor = 1.0  # 快衰落因子
    
    def calculate_channel_state(self, pos_a: Position, pos_b: Position, 
                               tx_node_type: str = 'vehicle', rx_node_type: str = 'rsu') -> ChannelState:
        """
        计算信道状态 - 3GPP标准式(11)-(16)
        
        Args:
            pos_a: 发送节点位置
            pos_b: 接收节点位置
            tx_node_type: 发送节点类型 ('vehicle', 'rsu', 'uav')
            rx_node_type: 接收节点类型 ('vehicle', 'rsu', 'uav')
            
        Returns:
            信道状态信息
        """
        # 1. 计算距离 - 论文式(10)
        distance = pos_a.distance_to(pos_b)
        
        # 2. 计算视距概率 - 3GPP标准式(11)
        los_probability = self._calculate_los_probability(distance)
        
        # 3. 计算路径损耗 - 3GPP标准式(12)-(13)
        path_loss_db = self._calculate_path_loss(distance, los_probability)
        
        # 4. 计算阴影衰落 - 随机变量
        shadowing_db = self._generate_shadowing(los_probability)
        
        # 5. 计算信道增益 - 3GPP标准式(14)
        channel_gain_linear = self._calculate_channel_gain(path_loss_db, shadowing_db, tx_node_type, rx_node_type)
        
        # 6. 计算干扰功率 (简化)
        interference_power = self._calculate_interference_power(pos_b)
        
        return ChannelState(
            distance=distance,
            los_probability=los_probability,
            path_loss_db=path_loss_db,
            shadowing_db=shadowing_db,
            channel_gain_linear=channel_gain_linear,
            interference_power=interference_power
        )
    
    def _calculate_los_probability(self, distance: float) -> float:
        """
        计算视距概率 - 对应论文式(11)
        P_LoS(d) = 1 if d ≤ d_0, exp(-(d-d_0)/α_LoS) if d > d_0
        """
        if distance <= self.los_threshold:
            return 1.0
        else:
            return math.exp(-(distance - self.los_threshold) / self.los_decay_factor)
    
    def _calculate_path_loss(self, distance: float, los_probability: float) -> float:
        """
        计算路径损耗 - 3GPP TS 38.901标准
        LoS: PL = 32.4 + 20*log10(fc) + 20*log10(d)
        NLoS: PL = 32.4 + 20*log10(fc) + 30*log10(d)
        其中 fc单位为GHz，d单位为km
        """
        # 确保距离至少为1米，避免log10(0)
        distance_km = max(distance / 1000.0, 0.001)
        frequency_ghz = self.carrier_frequency / 1e9
        
        # LoS路径损耗 - 3GPP标准式(12)
        los_path_loss = 32.4 + 20 * math.log10(frequency_ghz) + 20 * math.log10(distance_km)
        
        # NLoS路径损耗 - 3GPP标准式(13)
        nlos_path_loss = 32.4 + 20 * math.log10(frequency_ghz) + 30 * math.log10(distance_km)
        
        # 综合路径损耗
        combined_path_loss = los_probability * los_path_loss + (1 - los_probability) * nlos_path_loss
        
        return combined_path_loss
    
    def _generate_shadowing(self, los_probability: float) -> float:
        """生成阴影衰落"""
        if np.random.random() < los_probability:
            # LoS情况
            return np.random.normal(0, self.shadowing_std_los)
        else:
            # NLoS情况
            return np.random.normal(0, self.shadowing_std_nlos)
    
    def _calculate_channel_gain(self, path_loss_db: float, shadowing_db: float, 
                               tx_node_type: str = 'vehicle', rx_node_type: str = 'rsu') -> float:
        """
        计算信道增益 - 3GPP标准式(14)
        h = 10^(-L/10) * g_tx * g_rx * g_fading
        """
        # 根据节点类型选择天线增益
        tx_gain_map = {
            'vehicle': self.antenna_gain_vehicle,
            'rsu': self.antenna_gain_rsu,
            'uav': self.antenna_gain_uav
        }
        rx_gain_map = {
            'vehicle': self.antenna_gain_vehicle,
            'rsu': self.antenna_gain_rsu,
            'uav': self.antenna_gain_uav
        }
        
        tx_antenna_gain_db = tx_gain_map.get(tx_node_type, self.antenna_gain_vehicle)
        rx_antenna_gain_db = rx_gain_map.get(rx_node_type, self.antenna_gain_rsu)
        
        # 转换为线性值
        total_path_loss_db = path_loss_db + shadowing_db
        path_loss_linear = max(db_to_linear(total_path_loss_db), 1e-9)
        antenna_gain_linear = db_to_linear(tx_antenna_gain_db + rx_antenna_gain_db)
        
        # 总信道增益
        channel_gain = (antenna_gain_linear * self.fast_fading_factor) / path_loss_linear
        
        return channel_gain
    
    def _calculate_interference_power(self, receiver_pos: Position) -> float:
        """
        计算干扰功率 - 对应论文式(15)
        简化实现：基于位置的固定干扰模型
        """
        # 基础干扰功率
        base_interference = 1e-12  # W
        
        # 位置相关的干扰变化 (简化)
        interference_factor = 1.0 + 0.1 * math.sin(receiver_pos.x / 1000) * math.cos(receiver_pos.y / 1000)
        
        return base_interference * interference_factor
    
    def calculate_sinr(self, tx_power: float, channel_gain: float, 
                      interference_power: float, bandwidth: float) -> float:
        """
        计算信噪干扰比 - 3GPP标准式(16)
        SINR = (P_tx * h) / (I_ext + N_0 * B)
        其中 P_tx 以瓦特计，N_0 = -174 dBm/Hz (3GPP标准热噪声密度)
        """
        # 检查输入参数的有效性
        if tx_power <= 0 or channel_gain <= 0 or bandwidth <= 0:
            return 0.0
        
        signal_power = tx_power * channel_gain
        
        # 3GPP标准噪声功率计算: N = N_0 + 10*log10(B)
        # N_0 = -174 dBm/Hz, 转换为线性功率
        noise_power_dbm = self.thermal_noise_density + 10 * math.log10(bandwidth)
        noise_power_linear = dbm_to_watts(noise_power_dbm)
        
        total_interference_noise = interference_power + noise_power_linear
        
        # 防止除以零或过小值
        min_interference_noise = 1e-15  # 最小噪声功率
        if total_interference_noise <= min_interference_noise:
            total_interference_noise = min_interference_noise
        
        sinr_linear = signal_power / total_interference_noise
        
        # 限制SINR在合理范围内
        max_sinr = 1e6  # 防止过大值
        sinr_linear = min(sinr_linear, max_sinr)
        
        return sinr_linear
    
    def calculate_data_rate(self, sinr_linear: float, bandwidth: float) -> float:
        """
        计算传输速率 - 对应论文式(17)
        R = B * log2(1 + SINR) * η_coding
        """
        if sinr_linear <= 0:
            return 0.0
        
        rate = bandwidth * math.log2(1 + sinr_linear) * self.coding_efficiency
        return rate
    
    def calculate_transmission_delay(self, data_size: float, distance: float, 
                                   tx_power: float, bandwidth: float,
                                   pos_a: Position, pos_b: Position) -> Tuple[float, Dict]:
        """
        计算传输时延 - 对应论文式(18)
        T_trans = D/R + T_prop + T_proc
        
        Returns:
            (总时延, 详细信息字典)
        """
        # 1. 计算信道状态
        channel_state = self.calculate_channel_state(pos_a, pos_b)
        
        # 2. 计算SINR
        sinr_linear = self.calculate_sinr(tx_power, channel_state.channel_gain_linear,
                                        channel_state.interference_power, bandwidth)
        
        # 3. 计算数据速率
        data_rate = self.calculate_data_rate(sinr_linear, bandwidth)
        
        # 4. 计算各部分时延
        if data_rate > 0:
            transmission_delay = data_size / data_rate
        else:
            transmission_delay = float('inf')
        
        propagation_delay = distance / 3e8  # 光速传播
        total_delay = transmission_delay + propagation_delay + self.processing_delay
        
        # 详细信息
        details = {
            'channel_state': channel_state,
            'sinr_linear': sinr_linear,
            'sinr_db': linear_to_db(sinr_linear),
            'tx_power_watts': tx_power,
            'data_rate': data_rate,
            'transmission_delay': transmission_delay,
            'propagation_delay': propagation_delay,
            'processing_delay': self.processing_delay,
            'total_delay': total_delay
        }
        
        return total_delay, details


class ComputeEnergyModel:
    """
    计算能耗模型 - 对应论文第5.1节、第5.3节、第5.5节
    实现不同节点类型的计算能耗计算
    """
    
    def __init__(self):
        # 车辆能耗参数 - 论文式(5)-(9)
        self.vehicle_kappa1 = config.compute.vehicle_kappa1
        self.vehicle_kappa2 = config.compute.vehicle_kappa2
        self.vehicle_static_power = config.compute.vehicle_static_power
        self.vehicle_idle_power = config.compute.vehicle_idle_power
        
        # RSU能耗参数 - 论文式(20)-(21)
        # 🔧 修复：使用rsu_kappa而不是rsu_kappa2（避免混淆）
        self.rsu_kappa = getattr(config.compute, 'rsu_kappa', config.compute.rsu_kappa2)
        self.rsu_static_power = getattr(config.compute, 'rsu_static_power', 0.0)
        
        # UAV能耗参数 - 论文式(25)-(30)
        self.uav_kappa3 = config.compute.uav_kappa3
        self.uav_static_power = getattr(config.compute, 'uav_static_power', 0.0)
        self.uav_hover_power = config.compute.uav_hover_power
        
        # 并行处理效率
        self.parallel_efficiency = config.compute.parallel_efficiency
        self.time_slot_duration = getattr(config.network, 'time_slot_duration', 0.1)
    
    def calculate_vehicle_compute_energy(self, task: Task, cpu_frequency: float, 
                                       processing_time: float, time_slot_duration: float) -> Dict[str, float]:
        """
        计算车辆计算能耗 - 对应论文式(5)-(9)
        
        Returns:
            能耗详细信息字典
        """
        # 计算CPU利用率
        utilization = min(1.0, processing_time / time_slot_duration)
        
        # 动态功率模型 - 论文式(7)
        dynamic_power = (self.vehicle_kappa1 * (cpu_frequency ** 3) +
                        self.vehicle_kappa2 * (cpu_frequency ** 2) * utilization +
                        self.vehicle_static_power)
        
        # 计算能耗 - 论文式(8)
        active_time = processing_time
        idle_time = max(0, time_slot_duration - active_time)
        
        compute_energy = dynamic_power * active_time
        idle_energy = self.vehicle_idle_power * idle_time
        total_energy = compute_energy + idle_energy
        
        return {
            'dynamic_power': dynamic_power,
            'compute_energy': compute_energy,
            'idle_energy': idle_energy,
            'total_energy': total_energy,
            'utilization': utilization,
            'active_time': active_time,
            'idle_time': idle_time
        }
    
    def calculate_rsu_compute_energy(self, task: Task, cpu_frequency: float, 
                                   processing_time: float, is_active: bool = True) -> Dict[str, float]:
        """
        计算RSU计算能耗 - 对应论文式(20)-(22)
        
        Returns:
            能耗详细信息字典
        """
        if not is_active:
            return {
                'processing_power': 0.0,
                'processing_time': 0.0,
                'dynamic_energy': 0.0,
                'static_energy': 0.0,
                'accounted_time': 0.0,
                'compute_energy': 0.0,
                'total_energy': 0.0
            }
        
        # RSU处理功率 - 论文式(544): P = κ × f³
        # 🔧 修复：使用rsu_kappa而不是rsu_kappa2
        processing_power = self.rsu_kappa * (cpu_frequency ** 3)
        
        # 计算能耗
        dynamic_energy = processing_power * processing_time
        accounted_time = max(processing_time, self.time_slot_duration)
        static_energy = self.rsu_static_power * accounted_time
        total_energy = dynamic_energy + static_energy
        
        return {
            'processing_power': processing_power,
            'processing_time': processing_time,
            'dynamic_energy': dynamic_energy,
            'static_energy': static_energy,
            'accounted_time': accounted_time,
            'compute_energy': total_energy,
            'total_energy': total_energy
        }
    
    def calculate_uav_compute_energy(self, task: Task, cpu_frequency: float, 
                                   processing_time: float, battery_level: float = 1.0) -> Dict[str, float]:
        """
        计算UAV计算能耗 - 对应论文式(25)-(28)
        
        Returns:
            能耗详细信息字典
        """
        # 考虑电池电量对性能的影响
        battery_factor = max(0.5, battery_level)
        effective_frequency = cpu_frequency * battery_factor
        
        # UAV计算能耗 - 论文式(570): E = κ₃ × f³ × τ_active
        # 🔧 修复：从 f² × time 改为 f³ × time（与论文一致）
        processing_power = self.uav_kappa3 * (effective_frequency ** 3)
        dynamic_energy = processing_power * processing_time
        accounted_time = max(processing_time, self.time_slot_duration)
        static_energy = self.uav_static_power * accounted_time
        total_energy = dynamic_energy + static_energy
        
        return {
            'effective_frequency': effective_frequency,
            'battery_factor': battery_factor,
            'processing_time': processing_time,
            'dynamic_energy': dynamic_energy,
            'static_energy': static_energy,
            'accounted_time': accounted_time,
            'compute_energy': total_energy,
            'total_energy': total_energy
        }
    
    def calculate_uav_hover_energy(self, time_duration: float) -> Dict[str, float]:
        """
        计算UAV悬停能耗 - 对应论文式(29)-(30)
        
        Returns:
            悬停能耗信息字典
        """
        # 悬停能耗 - 论文式(29)-(30)简化版
        hover_energy = self.uav_hover_power * time_duration
        
        return {
            'hover_power': self.uav_hover_power,
            'hover_time': time_duration,
            'hover_energy': hover_energy,
            'total_energy': hover_energy
        }


class CommunicationEnergyModel:
    """
    通信能耗模型 - 对应论文式(19)和第5.5.1节
    计算无线传输的能耗
    """
    
    def __init__(self):
        # 传输功率参数（配置为 dBm，这里统一转换为瓦特以便计算能耗）
        self.vehicle_tx_power_dbm = config.communication.vehicle_tx_power
        self.rsu_tx_power_dbm = config.communication.rsu_tx_power
        self.uav_tx_power_dbm = config.communication.uav_tx_power
        self.vehicle_tx_power = dbm_to_watts(self.vehicle_tx_power_dbm)
        self.rsu_tx_power = dbm_to_watts(self.rsu_tx_power_dbm)
        self.uav_tx_power = dbm_to_watts(self.uav_tx_power_dbm)
        
        # 电路功率
        self.circuit_power = config.communication.circuit_power
        
        # 接收功率 (通常比发射功率小)
        self.rx_power_factor = 0.1  # 接收功率为发射功率的10%
    
    def calculate_transmission_energy(self, data_size: float, transmission_time: float, 
                                    node_type: str, include_circuit: bool = True) -> Dict[str, float]:
        """
        计算传输能耗 - 对应论文式(19)
        E^tx = P_tx * τ_tx + P_circuit * τ_active
        
        Args:
            data_size: 传输数据大小 (bits)
            transmission_time: 传输时间 (秒)
            node_type: 节点类型 ("vehicle", "rsu", "uav")
            include_circuit: 是否包含电路功耗
            
        Returns:
            传输能耗详细信息
        """
        # 获取发射功率（默认配置单位为 dBm，这里返回瓦特）
        if node_type == "vehicle":
            tx_power_dbm = self.vehicle_tx_power_dbm
            tx_power_watts = self.vehicle_tx_power
        elif node_type == "rsu":
            tx_power_dbm = self.rsu_tx_power_dbm
            tx_power_watts = self.rsu_tx_power
        elif node_type == "uav":
            tx_power_dbm = self.uav_tx_power_dbm
            tx_power_watts = self.uav_tx_power
        else:
            tx_power_dbm = self.vehicle_tx_power_dbm
            tx_power_watts = self.vehicle_tx_power  # 默认值
        
        # 传输能耗
        transmission_energy = tx_power_watts * transmission_time
        
        # 电路能耗
        if include_circuit:
            circuit_energy = self.circuit_power * transmission_time
        else:
            circuit_energy = 0.0
        
        total_energy = transmission_energy + circuit_energy
        
        return {
            'tx_power': tx_power_watts,
            'tx_power_dbm': tx_power_dbm,
            'transmission_time': transmission_time,
            'transmission_energy': transmission_energy,
            'circuit_energy': circuit_energy,
            'total_energy': total_energy,
            'data_size': data_size
        }
    
    def calculate_reception_energy(self, data_size: float, reception_time: float, 
                                 node_type: str) -> Dict[str, float]:
        """
        计算接收能耗 - 对应论文第5.5.1节
        
        Returns:
            接收能耗详细信息
        """
        # 获取对应的接收功率（默认配置是 dBm，这里使用瓦特）
        if node_type == "vehicle":
            tx_power_dbm = self.vehicle_tx_power_dbm
            base_power = self.vehicle_tx_power
        elif node_type == "rsu":
            tx_power_dbm = self.rsu_tx_power_dbm
            base_power = self.rsu_tx_power
        elif node_type == "uav":
            tx_power_dbm = self.uav_tx_power_dbm
            base_power = self.uav_tx_power
        else:
            tx_power_dbm = self.vehicle_tx_power_dbm
            base_power = self.vehicle_tx_power
        
        rx_power = base_power * self.rx_power_factor
        
        # 接收能耗
        reception_energy = rx_power * reception_time
        circuit_energy = self.circuit_power * reception_time
        
        total_energy = reception_energy + circuit_energy
        
        return {
            'rx_power': rx_power,
            'rx_power_dbm': tx_power_dbm + linear_to_db(self.rx_power_factor),
            'reception_time': reception_time,
            'reception_energy': reception_energy,
            'circuit_energy': circuit_energy,
            'total_energy': total_energy,
            'data_size': data_size
        }
    
    def calculate_communication_energy_total(self, task: Task, link_info: Dict, 
                                           tx_node_type: str, rx_node_type: str) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        计算完整通信过程的能耗 (发送+接收)
        
        Args:
            task: 任务信息
            link_info: 链路信息 (包含上传和下载时延)
            tx_node_type: 发送节点类型
            rx_node_type: 接收节点类型
            
        Returns:
            总通信能耗信息
        """
        # 上传能耗 (数据上传)
        upload_time = link_info.get('upload_transmission_time', link_info.get('upload_delay', 0.0))
        upload_tx_energy = self.calculate_transmission_energy(
            task.data_size, upload_time, tx_node_type)
        upload_rx_energy = self.calculate_reception_energy(
            task.data_size, upload_time, rx_node_type)
        
        # 下载能耗 (结果下载)
        download_time = link_info.get('download_transmission_time', link_info.get('download_delay', 0.0))
        download_tx_energy = self.calculate_transmission_energy(
            task.result_size, download_time, rx_node_type)
        download_rx_energy = self.calculate_reception_energy(
            task.result_size, download_time, tx_node_type)
        
        # 总能耗
        total_tx_energy = upload_tx_energy['total_energy'] + download_tx_energy['total_energy']
        total_rx_energy = upload_rx_energy['total_energy'] + download_rx_energy['total_energy']
        total_energy = total_tx_energy + total_rx_energy
        
        return {
            'upload_tx_energy': upload_tx_energy,
            'upload_rx_energy': upload_rx_energy,
            'download_tx_energy': download_tx_energy,
            'download_rx_energy': download_rx_energy,
            'total_tx_energy': total_tx_energy,
            'total_rx_energy': total_rx_energy,
            'total_communication_energy': total_energy
        }


class IntegratedCommunicationComputeModel:
    """
    集成通信计算模型
    整合论文第5节的所有通信和计算模型
    """
    
    def __init__(self):
        self.comm_model = WirelessCommunicationModel()
        self.compute_energy_model = ComputeEnergyModel()
        self.comm_energy_model = CommunicationEnergyModel()
    
    def evaluate_processing_option(self, task: Task, source_pos: Position, 
                                 target_pos: Position, target_node_info: Dict,
                                 processing_mode: str) -> Dict[str, Any]:
        """
        全面评估处理选项的时延和能耗
        
        Args:
            task: 待处理任务
            source_pos: 源节点位置
            target_pos: 目标节点位置  
            target_node_info: 目标节点信息
            processing_mode: 处理模式 ("local", "rsu", "uav")
            
        Returns:
            评估结果字典
        """
        results: Dict[str, Any] = {
            'total_delay': 0.0,
            'total_energy': 0.0,
            'communication_delay': 0.0,
            'computation_delay': 0.0,
            'communication_energy': 0.0,
            'computation_energy': 0.0
        }
        
        if processing_mode == "local":
            # 本地处理 - 无通信时延，只有计算
            cpu_freq = target_node_info.get('cpu_frequency', config.compute.vehicle_cpu_freq_range[1])
            processing_time = task.compute_cycles / (cpu_freq * self.compute_energy_model.parallel_efficiency)
            
            # 计算能耗
            energy_info = self.compute_energy_model.calculate_vehicle_compute_energy(
                task, cpu_freq, processing_time, config.network.time_slot_duration)
            
            results.update({
                'total_delay': processing_time,
                'computation_delay': processing_time,
                'total_energy': energy_info['total_energy'],
                'computation_energy': energy_info['total_energy']
            })
        
        elif processing_mode in ["rsu", "uav"]:
            # 远程处理 - 通信 + 计算
            
            # 1. 通信时延和能耗
            vehicle_tx_power_watts = dbm_to_watts(config.communication.vehicle_tx_power)
            upload_delay, upload_details = self.comm_model.calculate_transmission_delay(
                task.data_size, source_pos.distance_to(target_pos),
                vehicle_tx_power_watts,
                config.communication.total_bandwidth / 4,  # 分配带宽
                source_pos, target_pos
            )
            
            default_downlink_power_dbm = (config.communication.rsu_tx_power
                                          if processing_mode == "rsu"
                                          else config.communication.uav_tx_power)
            download_tx_power_dbm = target_node_info.get('tx_power', default_downlink_power_dbm)
            download_tx_power_watts = dbm_to_watts(download_tx_power_dbm)
            download_delay, download_details = self.comm_model.calculate_transmission_delay(
                task.result_size, source_pos.distance_to(target_pos),
                download_tx_power_watts,
                config.communication.total_bandwidth / 4,
                target_pos, source_pos
            )
            
            comm_delay = upload_delay + download_delay
            
            # 通信能耗
            link_info = {
                'upload_delay': upload_delay,
                'download_delay': download_delay,
                'upload_transmission_time': upload_details.get('transmission_delay', upload_delay),
                'download_transmission_time': download_details.get('transmission_delay', download_delay)
            }
            comm_energy_info = self.comm_energy_model.calculate_communication_energy_total(
                task, link_info, "vehicle", processing_mode)
            
            # 2. 计算时延和能耗
            cpu_freq = target_node_info.get('cpu_frequency', config.compute.rsu_cpu_freq)
            processing_time = task.compute_cycles / cpu_freq
            
            if processing_mode == "rsu":
                compute_energy_info = self.compute_energy_model.calculate_rsu_compute_energy(
                    task, cpu_freq, processing_time)
            else:  # uav
                battery_level = target_node_info.get('battery_level', 1.0)
                compute_energy_info = self.compute_energy_model.calculate_uav_compute_energy(
                    task, cpu_freq, processing_time, battery_level)
                
                # 添加悬停能耗
                total_time = comm_delay + processing_time
                hover_energy_info = self.compute_energy_model.calculate_uav_hover_energy(total_time)
                hover_energy = hover_energy_info['total_energy']
                compute_energy_info['hover_energy'] = hover_energy
                compute_energy_info['hover_details'] = hover_energy_info
                compute_energy_info['total_energy'] += hover_energy
                compute_energy_info['compute_energy'] = compute_energy_info.get('compute_energy', 0.0) + hover_energy
            
            # 汇总结果
            total_comm_energy = comm_energy_info['total_communication_energy']
            total_compute_energy = compute_energy_info['total_energy']
            
            # 确保能耗值是数值类型
            if isinstance(total_comm_energy, dict):
                total_comm_energy = 0.0  # 默认值，这种情况不应该出现
            if isinstance(total_compute_energy, dict):
                total_compute_energy = 0.0  # 默认值，这种情况不应该出现
            
            results.update({
                'total_delay': comm_delay + processing_time,
                'communication_delay': comm_delay,
                'computation_delay': processing_time,
                'total_energy': total_comm_energy + total_compute_energy,
                'communication_energy': total_comm_energy,
                'computation_energy': total_compute_energy,
                'upload_details': upload_details,
                'download_details': download_details,
                'comm_energy_details': comm_energy_info,
                'compute_energy_details': compute_energy_info
            })
        
        return results
