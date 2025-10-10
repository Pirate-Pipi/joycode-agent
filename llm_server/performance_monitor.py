#!/usr/bin/env python3
"""
LLM API Performance Monitoring Tool
Used to monitor and analyze performance issues of LLM API calls
"""

import time
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class APICallRecord:
    """API call record"""
    timestamp: str
    model_name: str
    base_url: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    attempt_count: int
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    status_code: Optional[int] = None


class PerformanceMonitor:
    """Performance monitor"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.records: List[APICallRecord] = []
        self.lock = threading.Lock()
        
        if log_file:
            self.log_file = Path(log_file)
        else:
            # Default log file path
            self.log_file = Path(__file__).parent / "performance_logs.jsonl"
        
        # Ensure the log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def record_call(self, record: APICallRecord):
        """Record an API call"""
        with self.lock:
            self.records.append(record)
            
            # 写入日志文件
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(asdict(record), ensure_ascii=False) + '\n')
            except Exception as e:
                logger.warning(f"Failed to write performance log: {e}")
    
    def get_statistics(self, hours: int = 24) -> Dict:
        """Get performance statistics"""
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        # Filter recent records
        recent_records = [r for r in self.records if r.start_time >= cutoff_time]
        
        if not recent_records:
            return {"message": "No records found in the specified time range"}
        
        # Compute statistics
        total_calls = len(recent_records)
        successful_calls = len([r for r in recent_records if r.success])
        failed_calls = total_calls - successful_calls
        
        durations = [r.duration for r in recent_records if r.success]
        avg_duration = sum(durations) / len(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        
        # Error type statistics
        error_types = {}
        for record in recent_records:
            if not record.success and record.error_type:
                error_types[record.error_type] = error_types.get(record.error_type, 0) + 1
        
        # Retry statistics
        retry_counts = [r.attempt_count for r in recent_records]
        avg_retries = sum(retry_counts) / len(retry_counts) if retry_counts else 0
        
        return {
            "time_range_hours": hours,
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 0,
            "average_duration": avg_duration,
            "max_duration": max_duration,
            "min_duration": min_duration,
            "average_retries": avg_retries,
            "error_types": error_types,
            "performance_issues": self._identify_performance_issues(recent_records)
        }
    
    def _identify_performance_issues(self, records: List[APICallRecord]) -> List[str]:
        """Identify performance issues"""
        issues = []
        
        if not records:
            return issues
        
        # Check for high failure rate
        success_rate = len([r for r in records if r.success]) / len(records)
        if success_rate < 0.8:
            issues.append(f"High failure rate: {(1-success_rate)*100:.1f}%")
        
        # Check for long response time
        successful_records = [r for r in records if r.success]
        if successful_records:
            avg_duration = sum(r.duration for r in successful_records) / len(successful_records)
            if avg_duration > 30:
                issues.append(f"Slow average response time: {avg_duration:.1f}s")
        
        # Check for frequent 504 errors
        gateway_timeouts = len([r for r in records if r.error_type and '504' in r.error_type])
        if gateway_timeouts > len(records) * 0.2:
            issues.append(f"Frequent gateway timeouts: {gateway_timeouts} out of {len(records)} calls")
        
        # Check for high retry rate
        high_retry_calls = len([r for r in records if r.attempt_count > 2])
        if high_retry_calls > len(records) * 0.3:
            issues.append(f"High retry rate: {high_retry_calls} calls required multiple retries")
        
        return issues
    
    def print_statistics(self, hours: int = 24):
        """Print performance statistics"""
        stats = self.get_statistics(hours)
        
        print(f"\n📊 LLM API performance statistics (last {hours} hours)")
        print("=" * 50)
        
        if "message" in stats:
            print(stats["message"])
            return
        
        print(f"Total calls: {stats['total_calls']}")
        print(f"Successful calls: {stats['successful_calls']}")
        print(f"Failed calls: {stats['failed_calls']}")
        print(f"Success rate: {stats['success_rate']*100:.1f}%")
        print(f"Average response time: {stats['average_duration']:.2f}s")
        print(f"Max response time: {stats['max_duration']:.2f}s")
        print(f"Min response time: {stats['min_duration']:.2f}s")
        print(f"Average retries: {stats['average_retries']:.1f}")
        
        if stats['error_types']:
            print("\n❌ Error types:")
            for error_type, count in stats['error_types'].items():
                print(f"   {error_type}: {count} times")
        
        if stats['performance_issues']:
            print("\n⚠️ Performance issues:")
            for issue in stats['performance_issues']:
                print(f"   • {issue}")
        
        print("=" * 50)


# Global monitor instance
_global_monitor = None
_monitor_lock = threading.Lock()


def get_monitor() -> PerformanceMonitor:
    """Get a global monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        with _monitor_lock:
            if _global_monitor is None:
                _global_monitor = PerformanceMonitor()
    return _global_monitor


def record_api_call(
    model_name: str,
    base_url: str,
    start_time: float,
    end_time: float,
    success: bool,
    attempt_count: int,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    status_code: Optional[int] = None
):
    """Record an API call (helper)"""
    record = APICallRecord(
        timestamp=datetime.now().isoformat(),
        model_name=model_name,
        base_url=base_url,
        start_time=start_time,
        end_time=end_time,
        duration=end_time - start_time,
        success=success,
        attempt_count=attempt_count,
        error_type=error_type,
        error_message=error_message,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        status_code=status_code
    )
    
    get_monitor().record_call(record)


def print_performance_stats(hours: int = 24):
    """Print performance statistics (helper)"""
    get_monitor().print_statistics(hours)

