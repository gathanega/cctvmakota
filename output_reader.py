"""
Output reader: single loader for output folder.
All decision-making data comes from files in output/ folder.
Traces every decision back to source file.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Dict, List

log = logging.getLogger("cctvmakota.output_reader")


@dataclass
class ModelMetrics:
    """Model performance from perbandingan_model.csv."""
    model_name: str
    dataset_name: str
    mAP50: float
    AP50_motorcycle: float
    AP50_car: float
    

class OutputReader:
    """Read and cache output folder files."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.cache: Dict = {}
        self.last_modified: Dict = {}
        
    def _check_modified(self, filename: str) -> bool:
        """Check if file has been modified since last read."""
        fpath = self.output_dir / filename
        if not fpath.exists():
            return False
        mtime = fpath.stat().st_mtime
        cached_mtime = self.last_modified.get(filename)
        if cached_mtime is None or mtime != cached_mtime:
            self.last_modified[filename] = mtime
            return True
        return False
    
    def read_model_comparison(self) -> Optional[List[ModelMetrics]]:
        """Read perbandingan_model.csv."""
        filename = "perbandingan_model.csv"
        if not self._check_modified(filename):
            return self.cache.get(filename)
        
        fpath = self.output_dir / filename
        if not fpath.exists():
            log.warning(f"File not found: {filename}")
            return None
        
        metrics = []
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Only keep balanced_r15 (selected model)
                    if row.get("model") == "balanced_r15":
                        metrics.append(ModelMetrics(
                            model_name=row.get("model"),
                            dataset_name=row.get("set_evaluasi"),
                            mAP50=float(row.get("mAP50", 0)),
                            AP50_motorcycle=float(row.get("AP50_motorcycle", 0)),
                            AP50_car=float(row.get("AP50_car", 0)),
                        ))
            self.cache[filename] = metrics
            log.info(f"Loaded {len(metrics)} model metrics from {filename}")
            return metrics
        except Exception as e:
            log.error(f"Failed to read {filename}: {e}")
            return None
    
    def read_dataset_stats(self) -> Optional[Dict]:
        """Read hasil_distribusi_dan_balancing.csv."""
        filename = "hasil_distribusi_dan_balancing.csv"
        if not self._check_modified(filename):
            return self.cache.get(filename)
        
        fpath = self.output_dir / filename
        if not fpath.exists():
            log.warning(f"File not found: {filename}")
            return None
        
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Return baseline row (only row in this file)
                    stats = {
                        "Dataset": row.get("Dataset"),
                        "Split": row.get("Split"),
                        "Jumlah_Gambar": int(row.get("Jumlah Gambar", 0)),
                        "Total_Instance": int(row.get("Total Instance", 0)),
                        "Car": int(row.get("Car", 0)),
                        "Motorcycle": int(row.get("Motorcycle", 0)),
                        "Bus": int(row.get("Bus", 0)),
                        "Truck": int(row.get("Truck", 0)),
                        "Imbalance_Ratio": float(row.get("Imbalance Ratio", 0)),
                        "Status": row.get("Status Balance"),
                    }
                    self.cache[filename] = stats
                    log.info(f"Loaded dataset stats from {filename}")
                    return stats
        except Exception as e:
            log.error(f"Failed to read {filename}: {e}")
            return None
        return None
    
    def read_analysis(self) -> Optional[Dict]:
        """Read analisis_model.json."""
        filename = "analisis_model.json"
        if not self._check_modified(filename):
            return self.cache.get(filename)
        
        fpath = self.output_dir / filename
        if not fpath.exists():
            log.warning(f"File not found: {filename}")
            return None
        
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.cache[filename] = data
            log.info(f"Loaded analysis from {filename}")
            return data
        except Exception as e:
            log.error(f"Failed to read {filename}: {e}")
            return None
    
    def get_decision_context(self) -> Dict:
        """Get all decision-supporting data."""
        return {
            "model_metrics": self.read_model_comparison(),
            "dataset_stats": self.read_dataset_stats(),
            "analysis": self.read_analysis(),
            "output_dir": str(self.output_dir),
            "timestamp": __import__("time").time(),
        }


def trace_decision(decision: str, output_reader: OutputReader) -> Dict:
    """Log a decision with full traceability."""
    context = output_reader.get_decision_context()
    return {
        "decision": decision,
        "timestamp": __import__("time").time(),
        "source_files": {
            "model": "perbandingan_model.csv",
            "dataset": "hasil_distribusi_dan_balancing.csv",
            "analysis": "analisis_model.json",
        },
        "context": context,
    }
