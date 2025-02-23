import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray,
                       data: pd.DataFrame) -> Dict:
        """错误分析"""
        try:
            # 确保输入长度一致并重置索引
            if not (len(y_true) == len(y_pred) == len(data)):
                logger.error(f"长度不匹配: y_true={len(y_true)}, y_pred={len(y_pred)}, data={len(data)}")
                raise ValueError("输入长度不一致")

            data = data.reset_index(drop=True)
            y_true = pd.Series(y_true).reset_index(drop=True)
            y_pred = pd.Series(y_pred).reset_index(drop=True)

            # 创建错误样本掩码
            error_mask = y_true != y_pred
            errors = data[error_mask].copy()

            # 基础错误统计
            error_analysis = {
                'error_by_type': errors['告警类型'].value_counts().to_dict(),
                'error_by_airport': errors.groupby(['起飞机场', '目的机场']).size().to_dict(),
                'error_rate': float(len(errors) / len(y_true) * 100),
                'false_positives': int(sum((y_true == 0) & (y_pred == 1))),
                'false_negatives': int(sum((y_true == 1) & (y_pred == 0)))
            }

            # 按小时统计
            hourly_stats = pd.DataFrame({
                'total': data.groupby('alarm_hour').size(),
                'errors': errors.groupby('alarm_hour').size()
            }).fillna(0)

            hourly_stats['error_rate'] = (
                    hourly_stats['errors'] / hourly_stats['total'] * 100
            ).round(2)

            error_analysis['error_by_hour'] = {
                hour: {
                    'total_samples': int(row['total']),
                    'error_count': int(row['errors']),
                    'error_rate': float(row['error_rate'])
                }
                for hour, row in hourly_stats.iterrows()
            }

            # 年月维度分析
            yearly_monthly_stats = {}
            data['year'] = pd.to_datetime(data['告警日期']).dt.year
            data['month'] = pd.to_datetime(data['告警日期']).dt.month
            errors['year'] = pd.to_datetime(errors['告警日期']).dt.year
            errors['month'] = pd.to_datetime(errors['告警日期']).dt.month

            for year in sorted(data['year'].unique()):
                yearly_monthly_stats[year] = {}
                year_data = data[data['year'] == year]
                year_errors = errors[errors['year'] == year]

                for month in sorted(year_data['month'].unique()):
                    month_data = year_data[year_data['month'] == month]
                    month_errors = year_errors[year_errors['month'] == month]

                    # 获取月份数据的索引
                    month_indices = month_data.index
                    month_mask = error_mask[month_indices]

                    # 计算虚报和漏报
                    month_true = y_true[month_indices]
                    month_pred = y_pred[month_indices]
                    false_positives = sum((month_true == 0) & (month_pred == 1))
                    false_negatives = sum((month_true == 1) & (month_pred == 0))

                    yearly_monthly_stats[year][month] = {
                        '样本数': len(month_data),
                        '预测错误数': len(month_errors),
                        '错误率': f"{(len(month_errors) / len(month_data) * 100):.2f}%",
                        '虚报数': int(false_positives),
                        '漏报数': int(false_negatives),
                        '告警类型错误分布': month_errors['告警类型'].value_counts().to_dict(),
                        '小时错误分布': month_errors.groupby('alarm_hour').size().to_dict()
                    }

            error_analysis['yearly_monthly_stats'] = yearly_monthly_stats
            return error_analysis

        except Exception as e:
            logger.error(f"错误分析失败: {e}")
            return {
                'error': str(e),
                'error_rate': 0,
                'false_positives': 0,
                'false_negatives': 0
            }

    def _basic_error_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                              errors: pd.DataFrame) -> Dict:
        """基础错误统计"""
        return {
            'error_by_type': errors['告警类型'].value_counts().to_dict(),
            'error_by_airport': errors.groupby(['起飞机场', '目的机场']).size().to_dict(),
            'error_rate': float(len(errors) / len(y_true) * 100),
            'false_positives': int(sum((y_true == 0) & (y_pred == 1))),
            'false_negatives': int(sum((y_true == 1) & (y_pred == 0)))
        }

    def _temporal_error_analysis(self, data: pd.DataFrame, errors: pd.DataFrame,
                                 error_mask: np.ndarray) -> Dict:
        """时间维度错误分析"""
        # 按小时统计
        hourly_stats = pd.DataFrame({
            'total': data.groupby('alarm_hour').size(),
            'errors': errors.groupby('alarm_hour').size()
        }).fillna(0)

        hourly_stats['error_rate'] = (
                hourly_stats['errors'] / hourly_stats['total'] * 100
        ).round(2)

        # 构建年月统计
        yearly_monthly_stats = {}
        for year in sorted(data['alarm_year'].unique()):
            yearly_monthly_stats[year] = self._analyze_year(
                data, error_mask, year
            )

        return {
            'error_by_hour': self._format_hourly_stats(hourly_stats),
            'yearly_monthly_stats': yearly_monthly_stats
        }

    def _analyze_year(self, data: pd.DataFrame, error_mask: np.ndarray,
                      year: int) -> Dict:
        """分析特定年份的错误"""
        year_data = data[data['alarm_year'] == year]
        year_stats = {}

        for month in sorted(year_data['alarm_month'].unique()):
            month_mask = (year_data['alarm_month'] == month)
            month_data = year_data[month_mask]
            month_errors = month_data[error_mask[month_data.index]]

            year_stats[month] = {
                '样本数': len(month_data),
                '预测错误数': len(month_errors),
                '错误率': f'{(len(month_errors) / len(month_data) * 100):.2f}%',
                '告警类型错误分布': month_errors['告警类型'].value_counts().to_dict(),
                '小时错误分布': month_errors.groupby('alarm_hour').size().to_dict()
            }

        return year_stats

    def _format_hourly_stats(self, hourly_stats: pd.DataFrame) -> Dict:
        """格式化小时统计数据"""
        return {
            hour: {
                'total_samples': int(row['total']),
                'error_count': int(row['errors']),
                'error_rate': float(row['error_rate'])
            }
            for hour, row in hourly_stats.iterrows()
        }

    def _get_error_examples(self, errors: pd.DataFrame, y_true: np.ndarray,
                            y_pred: np.ndarray) -> Dict:
        """获取错误样本示例"""
        if errors.empty:
            return {}

        sample_errors = errors.sample(min(5, len(errors)))
        return {
            idx: {
                '告警类型': row['告警类型'],
                '航班号': row['航班号三字码'],
                '起飞机场': row['起飞机场'],
                '目的机场': row['目的机场'],
                '真实标签': int(y_true[idx]),
                '预测标签': int(y_pred[idx]),
                '告警描述': row.get('告警描述', 'N/A'),
                '告警时间': row.get('告警时间', 'N/A')
            }
            for idx, row in sample_errors.iterrows()
        }

    def _get_error_analysis_default(self) -> Dict:
        """返回默认的错误分析结果"""
        return {
            'error': '错误分析失败',
            'error_rate': -1,
            'false_positives': -1,
            'false_negatives': -1
        }

    def print_evaluation_results(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 error_analysis: Dict) -> None:
        """打印评估结果"""
        print("\n=== 评估结果 ===")
        print("\n分类报告:")
        print(classification_report(y_true, y_pred, zero_division=1))

        print("\n混淆矩阵:")
        print(confusion_matrix(y_true, y_pred))

        print("\n错误统计:")
        print(f"错误率: {error_analysis['error_rate']:.2f}%")
        print(f"虚报数: {error_analysis['false_positives']}")
        print(f"漏报数: {error_analysis['false_negatives']}")

        if 'error_by_type' in error_analysis:
            print("\n告警类型错误分布:")
            for alarm_type, count in error_analysis['error_by_type'].items():
                print(f"  {alarm_type}: {count}次")

        # 打印小时维度错误分布
        if 'error_by_hour' in error_analysis:
            print("\n小时维度错误分布:")
            for hour, stats in sorted(error_analysis['error_by_hour'].items()):
                print(f"  {hour}时:")
                print(f"    样本总数: {stats['total_samples']}")
                print(f"    错误数: {stats['error_count']}")
                print(f"    错误率: {stats['error_rate']:.2f}%")

        # 打印年度月度分析结果
        if 'yearly_monthly_stats' in error_analysis:
            print("\n=== 年度月度预测结果分析 ===")
            yearly_monthly_stats = error_analysis['yearly_monthly_stats']
            for year in sorted(yearly_monthly_stats.keys()):
                print(f"\n=== {year}年 ===")
                print("-" * 50)
                for month in sorted(yearly_monthly_stats[year].keys()):
                    stats = yearly_monthly_stats[year][month]
                    print(f"\n{year}年{month}月统计:")
                    print(f"总样本数: {stats['样本数']}")
                    print(f"预测错误数: {stats['预测错误数']}")
                    print(f"错误率: {stats['错误率']}")
                    print(f"虚报数: {stats['虚报数']}")
                    print(f"漏报数: {stats['漏报数']}")

                    print("告警类型错误分布:")
                    if isinstance(stats['告警类型错误分布'], dict):
                        for alarm_type, count in stats['告警类型错误分布'].items():
                            print(f"  - {alarm_type}: {count}次")
                    else:
                        print(f"  {stats['告警类型错误分布']}")

                    print("\n小时错误分布:")
                    if stats['小时错误分布']:
                        for hour, count in sorted(stats['小时错误分布'].items()):
                            print(f"  - {hour}时: {count}次")