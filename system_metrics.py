import os
import logging

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)

def get_cpu_temp() -> float:
    """Read CPU / SoC temperature on Raspberry Pi or Linux."""
    try:
        if HAS_PSUTIL and hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if "cpu_thermal" in temps and temps["cpu_thermal"]:
                return round(temps["cpu_thermal"][0].current, 1)
            if "coretemp" in temps and temps["coretemp"]:
                return round(temps["coretemp"][0].current, 1)
        
        # Linux / RPi sysfs thermal fallback
        thermal_path = "/sys/class/thermal/thermal_zone0/temp"
        if os.path.exists(thermal_path):
            with open(thermal_path, "r") as f:
                temp_raw = float(f.read().strip())
                return round(temp_raw / 1000.0, 1)
    except Exception as e:
        logger.debug(f"Error reading CPU temp: {e}")
    return 0.0

def get_system_metrics() -> dict:
    """Fetch real-time CPU %, RAM %, Temperature, and Disk % telemetry."""
    metrics = {
        "cpu_percent": 0.0,
        "ram_percent": 0.0,
        "ram_used_mb": 0,
        "ram_total_mb": 0,
        "cpu_temp": get_cpu_temp(),
        "disk_percent": 0.0
    }

    if HAS_PSUTIL:
        try:
            metrics["cpu_percent"] = round(psutil.cpu_percent(interval=None), 1)
            mem = psutil.virtual_memory()
            metrics["ram_percent"] = round(mem.percent, 1)
            metrics["ram_used_mb"] = int(mem.used / (1024 * 1024))
            metrics["ram_total_mb"] = int(mem.total / (1024 * 1024))
            disk = psutil.disk_usage("/")
            metrics["disk_percent"] = round(disk.percent, 1)
        except Exception as e:
            logger.debug(f"Error reading psutil metrics: {e}")

    return metrics
