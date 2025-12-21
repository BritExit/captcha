import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'wqy-microhei', 'NotoSansCJK-Regular']  # 中文字体列表

def analyze_errors_with_custom_paths(csv_path, output_dir="./error_analysis_results"):
    """
    完整错误分析流程
    
    Args:
        csv_path: 输入CSV文件路径
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建完整输出路径
    summary_path = os.path.join(output_dir, 'error_summary.csv')
    plot_path = os.path.join(output_dir, 'error_analysis.png')
    confusion_path = os.path.join(output_dir, 'char_confusion_matrix.png')
    detailed_path = os.path.join(output_dir, 'detailed_errors.csv')
    
    print(f"开始分析错误...")
    print(f"输入文件: {csv_path}")
    print(f"输出目录: {output_dir}")
    print(f"输出文件:")
    print(f"  - 总结报告: {summary_path}")
    print(f"  - 分析图表: {plot_path}")
    print(f"  - 混淆矩阵: {confusion_path}")
    print(f"  - 详细数据: {detailed_path}")
    print("-" * 60)
    
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 创建错误类型列
    df['color_error'] = df['pred_color'] != df['true_color']
    df['char_error'] = df['pred_char'] != df['true_char']
    df['error_type'] = 'both_correct'
    
    # 分类错误类型
    df.loc[df['color_error'] & ~df['char_error'], 'error_type'] = 'color_only'
    df.loc[~df['color_error'] & df['char_error'], 'error_type'] = 'char_only'
    df.loc[df['color_error'] & df['char_error'], 'error_type'] = 'both_wrong'
    
    # 计算总体统计
    total_errors = len(df)
    color_errors = df['color_error'].sum()
    char_errors = df['char_error'].sum()
    both_wrong = (df['color_error'] & df['char_error']).sum()
    
    print("错误分析报告")
    print("=" * 60)
    print(f"总错误样本数: {total_errors}")
    print(f"颜色错误数: {color_errors} ({color_errors/total_errors*100:.1f}%)")
    print(f"字符错误数: {char_errors} ({char_errors/total_errors*100:.1f}%)")
    print(f"颜色字符都错: {both_wrong} ({both_wrong/total_errors*100:.1f}%)")
    print(f"仅颜色错误: {(df['error_type'] == 'color_only').sum()}")
    print(f"仅字符错误: {(df['error_type'] == 'char_only').sum()}")
    
    # 错误类型分布
    error_type_counts = df['error_type'].value_counts()
    print("\n错误类型分布:")
    for error_type, count in error_type_counts.items():
        print(f"  {error_type}: {count} ({count/total_errors*100:.1f}%)")
    
    # 字符错误分析
    char_error_df = df[df['char_error']]
    if not char_error_df.empty:
        print("\n字符错误分析:")
        print("-" * 40)
        
        # 统计每个真实字符的出错次数
        char_error_counts = char_error_df['true_char'].value_counts().sort_index()
        print("各字符出错次数:")
        for char, count in char_error_counts.items():
            print(f"  字符 '{char}': {count} 次错误")
        
        # 统计最常见的错误对
        confusion_data = []
        for _, row in char_error_df.iterrows():
            confusion_data.append([row['true_char'], row['pred_char']])
        
        confusion_df = pd.DataFrame(confusion_data, columns=['true_char', 'pred_char'])
        confusion_counts = confusion_df.groupby(['true_char', 'pred_char']).size().reset_index(name='count')
        confusion_counts = confusion_counts.sort_values('count', ascending=False)
        
        print("\n最常见的混淆对:")
        for _, row in confusion_counts.head(10).iterrows():
            print(f"  '{row['true_char']}' 被误认为 '{row['pred_char']}': {row['count']} 次")
        
        # 创建字符混淆热力图
        create_confusion_heatmap(confusion_df, confusion_path)
    
    # 颜色错误分析
    color_error_df = df[df['color_error']]
    if not color_error_df.empty:
        print("\n颜色错误分析:")
        print("-" * 40)
        
        # 颜色混淆情况
        color_confusions = []
        for _, row in color_error_df.iterrows():
            color_confusions.append(f"{row['true_color']}→{row['pred_color']}")
        
        color_confusion_counts = Counter(color_confusions)
        
        print("颜色混淆情况:")
        for confusion, count in color_confusion_counts.most_common():
            print(f"  {confusion}: {count} 次")
    
    # 创建可视化图表
    create_visualizations(df, plot_path)
    
    # 生成总结报告
    generate_summary_report(df, summary_path)
    
    # 保存详细的错误数据
    df.to_csv(detailed_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print(f"分析完成！文件已保存到: {output_dir}")
    
    return df

def create_confusion_heatmap(confusion_df, output_path):
    """创建字符混淆热力图"""
    confusion_matrix = pd.crosstab(
        confusion_df['true_char'], 
        confusion_df['pred_char']
    )
    
    if not confusion_matrix.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('字符混淆矩阵')
        plt.xlabel('predict char')
        plt.ylabel('true char')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def create_visualizations(df, output_path):
    """创建可视化图表"""
    plt.figure(figsize=(12, 4))
    
    # 1. 错误类型分布饼图
    plt.subplot(1, 3, 1)
    error_type_counts = df['error_type'].value_counts()
    colors = ['#4CAF50', '#FF9800', '#F44336', '#2196F3']
    plt.pie(error_type_counts.values, labels=error_type_counts.index, 
            autopct='%1.1f%%', colors=colors[:len(error_type_counts)])
    plt.title('错误类型分布')
    
    # 2. 字符错误条形图
    plt.subplot(1, 3, 2)
    char_error_df = df[df['char_error']]
    if not char_error_df.empty:
        char_errors_by_char = char_error_df['true_char'].value_counts().sort_index()
        plt.bar(char_errors_by_char.index, char_errors_by_char.values)
        plt.xlabel('字符')
        plt.ylabel('错误次数')
        plt.title('各字符错误次数')
        plt.xticks(rotation=45)
    
    # 3. 颜色错误分析
    plt.subplot(1, 3, 3)
    color_error_df = df[df['color_error']]
    if not color_error_df.empty:
        color_confusions = []
        for _, row in color_error_df.iterrows():
            color_confusions.append(f"{row['true_color']}→{row['pred_color']}")
        
        color_confusion_counts = Counter(color_confusions)
        
        labels = list(color_confusion_counts.keys())
        values = list(color_confusion_counts.values())
        
        plt.bar(range(len(labels)), values)
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.xlabel('颜色混淆')
        plt.ylabel('次数')
        plt.title('颜色混淆情况')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(df, output_path):
    """生成详细的错误总结表"""
    summary_data = []
    
    # 按错误类型分组统计
    for error_type, group in df.groupby('error_type'):
        summary_data.append({
            '错误类型': error_type,
            '样本数量': len(group),
            '占比': f"{len(group)/len(df)*100:.1f}%"
        })
    
    # 字符错误详细统计
    char_error_df = df[df['char_error']]
    if not char_error_df.empty:
        char_stats = char_error_df['true_char'].value_counts()
        for char, count in char_stats.items():
            summary_data.append({
                '错误类型': f"字符'{char}'错误",
                '样本数量': count,
                '占比': f"{count/len(df)*100:.1f}%"
            })
    
    # 颜色错误详细统计
    color_error_df = df[df['color_error']]
    if not color_error_df.empty:
        color_stats = color_error_df.groupby(['true_color', 'pred_color']).size().reset_index(name='count')
        for _, row in color_stats.iterrows():
            summary_data.append({
                '错误类型': f"颜色{row['true_color']}→{row['pred_color']}",
                '样本数量': row['count'],
                '占比': f"{row['count']/len(df)*100:.1f}%"
            })
    
    # 保存总结报告
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return summary_df

# 使用示例
if __name__ == "__main__":
    # 示例: 自定义输出目录
    csv_file = "./error_results/error_predictions.csv"
    output_directory = "./error_analysis_results"
    analyze_errors_with_custom_paths(csv_file, output_directory)