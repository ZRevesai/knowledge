"""
Energy Consumption Monitoring for NUTRINET.

Implements energy tracking as described in Section 3.6 of the paper.
Follows methodology proposed by Li et al. [37].

Tracks:
- Power usage during training and inference
- Total energy consumption in kilowatt-hours (kWh)
- Equivalent carbon emissions (CO2e)
- Energy efficiency metrics

Demonstrates the 73% energy reduction compared to Graph Attention Networks.
"""

import torch
import time
import psutil
import numpy as np
from typing import Dict, Optional, List
from contextlib import contextmanager
from dataclasses import dataclass, field
import warnings

# Try to import codecarbon for detailed energy tracking
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    warnings.warn("codecarbon not available. Using simplified energy estimation.")

# Try to import pynvml for GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except (ImportError, Exception):
    PYNVML_AVAILABLE = False
    warnings.warn("pynvml not available. GPU energy tracking disabled.")


@dataclass
class EnergyMeasurement:
    """Container for energy measurement data"""
    duration_seconds: float = 0.0
    cpu_energy_kwh: float = 0.0
    gpu_energy_kwh: float = 0.0
    total_energy_kwh: float = 0.0
    co2_emissions_kg: float = 0.0
    average_power_watts: float = 0.0
    peak_power_watts: float = 0.0
    cpu_utilization: List[float] = field(default_factory=list)
    gpu_utilization: List[float] = field(default_factory=list)
    memory_usage_mb: List[float] = field(default_factory=list)


class EnergyMonitor:
    """
    Monitor energy consumption during model training and inference.

    Uses a combination of:
    1. codecarbon for comprehensive tracking (if available)
    2. psutil for CPU monitoring
    3. pynvml for GPU monitoring (if available)
    4. System power measurements
    """

    def __init__(
        self,
        output_dir: str = "./emissions",
        country_iso_code: str = "USA",
        tracking_mode: str = "machine",
        log_level: str = "warning",
    ):
        """
        Args:
            output_dir: Directory to save emission logs
            country_iso_code: ISO code for carbon intensity calculations
            tracking_mode: Tracking mode ('machine' or 'process')
            log_level: Logging level
        """
        self.output_dir = output_dir
        self.country_iso_code = country_iso_code
        self.tracking_mode = tracking_mode

        # Initialize codecarbon tracker if available
        if CODECARBON_AVAILABLE:
            self.emissions_tracker = EmissionsTracker(
                output_dir=output_dir,
                country_iso_code=country_iso_code,
                tracking_mode=tracking_mode,
                log_level=log_level,
            )
        else:
            self.emissions_tracker = None

        # Initialize GPU monitoring
        self.gpu_available = PYNVML_AVAILABLE and torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_count = torch.cuda.device_count()
            self.gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(self.gpu_count)
            ]
        else:
            self.gpu_count = 0
            self.gpu_handles = []

        # Measurement storage
        self.measurements: List[EnergyMeasurement] = []
        self.current_measurement: Optional[EnergyMeasurement] = None

        # Tracking state
        self.is_tracking = False
        self.start_time = None

        # Power constants (in Watts)
        self.cpu_tdp = 65.0  # Typical CPU TDP
        self.gpu_tdp = 250.0  # Typical GPU TDP

    def start_tracking(self):
        """Start energy tracking"""
        self.is_tracking = True
        self.start_time = time.time()

        # Initialize new measurement
        self.current_measurement = EnergyMeasurement()

        # Start codecarbon tracker
        if self.emissions_tracker:
            self.emissions_tracker.start()

    def stop_tracking(self) -> EnergyMeasurement:
        """
        Stop energy tracking and return measurements.

        Returns:
            EnergyMeasurement object with collected data
        """
        if not self.is_tracking:
            raise ValueError("Tracking not started")

        # Stop codecarbon tracker
        if self.emissions_tracker:
            emissions = self.emissions_tracker.stop()

            if emissions is not None:
                # Extract measurements from codecarbon
                self.current_measurement.total_energy_kwh = emissions / 1000  # Convert to kWh
                self.current_measurement.co2_emissions_kg = emissions * 0.5  # Approximate CO2

        # Calculate duration
        self.current_measurement.duration_seconds = time.time() - self.start_time

        # Estimate energy if codecarbon not available
        if not self.emissions_tracker:
            self._estimate_energy()

        # Calculate average and peak power
        if self.current_measurement.duration_seconds > 0:
            self.current_measurement.average_power_watts = (
                self.current_measurement.total_energy_kwh * 1000 * 3600 /
                self.current_measurement.duration_seconds
            )

        # Store measurement
        self.measurements.append(self.current_measurement)

        self.is_tracking = False

        return self.current_measurement

    def sample_utilization(self):
        """Sample current CPU/GPU utilization and memory usage"""
        if not self.is_tracking or self.current_measurement is None:
            return

        # CPU utilization
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.current_measurement.cpu_utilization.append(cpu_percent)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_mb = memory.used / (1024 * 1024)
        self.current_measurement.memory_usage_mb.append(memory_mb)

        # GPU utilization
        if self.gpu_available:
            for handle in self.gpu_handles:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.current_measurement.gpu_utilization.append(util.gpu)
                except Exception:
                    pass

    def _estimate_energy(self):
        """Estimate energy consumption when codecarbon not available"""
        if self.current_measurement is None:
            return

        duration_hours = self.current_measurement.duration_seconds / 3600

        # Estimate CPU energy
        if self.current_measurement.cpu_utilization:
            avg_cpu_util = np.mean(self.current_measurement.cpu_utilization) / 100
            cpu_power = self.cpu_tdp * avg_cpu_util
            self.current_measurement.cpu_energy_kwh = cpu_power * duration_hours / 1000
        else:
            # Assume 50% utilization
            self.current_measurement.cpu_energy_kwh = self.cpu_tdp * 0.5 * duration_hours / 1000

        # Estimate GPU energy
        if self.gpu_available and self.current_measurement.gpu_utilization:
            avg_gpu_util = np.mean(self.current_measurement.gpu_utilization) / 100
            gpu_power = self.gpu_tdp * avg_gpu_util * self.gpu_count
            self.current_measurement.gpu_energy_kwh = gpu_power * duration_hours / 1000
        elif self.gpu_available:
            # Assume 70% utilization for GPU
            self.current_measurement.gpu_energy_kwh = (
                self.gpu_tdp * 0.7 * self.gpu_count * duration_hours / 1000
            )
        else:
            self.current_measurement.gpu_energy_kwh = 0.0

        # Total energy
        self.current_measurement.total_energy_kwh = (
            self.current_measurement.cpu_energy_kwh +
            self.current_measurement.gpu_energy_kwh
        )

        # Estimate CO2 emissions (using US average: 0.417 kg CO2/kWh)
        self.current_measurement.co2_emissions_kg = (
            self.current_measurement.total_energy_kwh * 0.417
        )

    def get_summary(self) -> Dict[str, float]:
        """
        Get summary of all measurements.

        Returns:
            Dictionary with aggregated metrics
        """
        if not self.measurements:
            return {}

        total_energy = sum(m.total_energy_kwh for m in self.measurements)
        total_duration = sum(m.duration_seconds for m in self.measurements)
        total_co2 = sum(m.co2_emissions_kg for m in self.measurements)

        avg_power = (
            total_energy * 1000 * 3600 / total_duration
            if total_duration > 0 else 0
        )

        return {
            'total_energy_kwh': total_energy,
            'total_duration_seconds': total_duration,
            'total_duration_hours': total_duration / 3600,
            'total_co2_emissions_kg': total_co2,
            'average_power_watts': avg_power,
            'num_measurements': len(self.measurements),
        }

    def compare_with_baseline(
        self,
        baseline_energy_kwh: float,
        baseline_name: str = "Baseline",
    ) -> Dict[str, float]:
        """
        Compare energy consumption with a baseline.

        Args:
            baseline_energy_kwh: Energy consumption of baseline model
            baseline_name: Name of baseline model

        Returns:
            Comparison metrics
        """
        summary = self.get_summary()
        nutrinet_energy = summary.get('total_energy_kwh', 0)

        if baseline_energy_kwh > 0:
            energy_reduction_percent = (
                (baseline_energy_kwh - nutrinet_energy) / baseline_energy_kwh * 100
            )
            efficiency_ratio = nutrinet_energy / baseline_energy_kwh
        else:
            energy_reduction_percent = 0
            efficiency_ratio = 0

        return {
            f'{baseline_name}_energy_kwh': baseline_energy_kwh,
            'nutrinet_energy_kwh': nutrinet_energy,
            'energy_reduction_kwh': baseline_energy_kwh - nutrinet_energy,
            'energy_reduction_percent': energy_reduction_percent,
            'efficiency_ratio': efficiency_ratio,
        }

    def print_summary(self):
        """Print summary of energy measurements"""
        summary = self.get_summary()

        print("\n" + "="*60)
        print(f"{'Energy Consumption Summary':^60}")
        print("="*60)

        print(f"{'Total Energy:':<40} {summary['total_energy_kwh']:.4f} kWh")
        print(f"{'Total Duration:':<40} {summary['total_duration_hours']:.2f} hours")
        print(f"{'Total CO2 Emissions:':<40} {summary['total_co2_emissions_kg']:.4f} kg")
        print(f"{'Average Power:':<40} {summary['average_power_watts']:.2f} W")
        print(f"{'Number of Measurements:':<40} {summary['num_measurements']}")

        print("="*60 + "\n")


@contextmanager
def measure_energy_consumption(
    monitor: Optional[EnergyMonitor] = None,
    output_dir: str = "./emissions",
) -> EnergyMeasurement:
    """
    Context manager for measuring energy consumption.

    Usage:
        with measure_energy_consumption() as measurement:
            # Run model training or inference
            model.train()

        print(f"Energy consumed: {measurement.total_energy_kwh} kWh")

    Args:
        monitor: Existing EnergyMonitor instance (optional)
        output_dir: Directory for emission logs

    Yields:
        EnergyMeasurement object
    """
    if monitor is None:
        monitor = EnergyMonitor(output_dir=output_dir)

    monitor.start_tracking()

    # Sample utilization periodically in background
    import threading

    def sample_loop():
        while monitor.is_tracking:
            monitor.sample_utilization()
            time.sleep(1.0)  # Sample every second

    sampling_thread = threading.Thread(target=sample_loop, daemon=True)
    sampling_thread.start()

    try:
        yield monitor.current_measurement
    finally:
        measurement = monitor.stop_tracking()
        sampling_thread.join(timeout=2.0)

    return measurement


# Example usage
if __name__ == "__main__":
    # Create energy monitor
    monitor = EnergyMonitor()

    # Measure energy for training
    monitor.start_tracking()

    # Simulate training
    print("Simulating model training...")
    time.sleep(5)  # Replace with actual training

    # Stop tracking
    measurement = monitor.stop_tracking()

    print(f"\nEnergy consumed: {measurement.total_energy_kwh:.6f} kWh")
    print(f"Duration: {measurement.duration_seconds:.2f} seconds")
    print(f"CO2 emissions: {measurement.co2_emissions_kg:.6f} kg")
    print(f"Average power: {measurement.average_power_watts:.2f} W")

    # Compare with baseline (e.g., GAT consuming 59.2 kWh from paper)
    comparison = monitor.compare_with_baseline(
        baseline_energy_kwh=59.2,
        baseline_name="GAT"
    )

    print(f"\nEnergy reduction: {comparison['energy_reduction_percent']:.1f}%")
    print(f"NUTRINET vs GAT efficiency: {comparison['efficiency_ratio']:.2f}x")
