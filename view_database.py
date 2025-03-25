import os
import sqlite3
import pandas as pd
import json

# 定义数据库文件路径
DB_PATH = './data/ideabench.db'

def load_and_display_database():
    """
    加载SQLite数据库并以DataFrame形式展示内容
    """
    # 检查数据库文件是否存在
    if not os.path.exists(DB_PATH):
        print(f"错误: 数据库文件 '{DB_PATH}' 不存在")
        return
    
    # 创建到数据库的连接
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # 查询所有结果
        query = "SELECT * FROM results ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn)
        
        # 处理JSON字段
        for json_col in ['parsed_scores', 'parsed_reasoning']:
            if json_col in df.columns:
                df[json_col] = df[json_col].apply(
                    lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else x
                )
        
        # 打印基本统计信息
        print(f"数据库中共有 {len(df)} 条记录")
        print("\n基本统计信息:")
        print(f"唯一关键词: {df['keywords'].nunique()} 个")
        print(f"想法模型种类: {df['idea_model'].nunique()} 个")
        print(f"评价模型种类: {df['critic_model'].nunique()} 个")
        
        # 显示DataFrame的概览
        print("\n数据预览:")
        # 选择更有意义的列进行显示
        display_columns = [
            'id', 'timestamp', 'keywords', 'idea_model', 'critic_model', 
            'parsed_scores', 'first_was_rejected'
        ]
        preview_df = df[display_columns].head(10)
        
        # 限制idea和critique的长度，以便更好地显示
        pd.set_option('display.max_colwidth', 50)
        
        # 打印DataFrame
        print(preview_df)
        
        # 还原pandas显示设置
        pd.reset_option('display.max_colwidth')
        
        return df
    
    except sqlite3.Error as e:
        print(f"读取数据库时出错: {e}")
    
    finally:
        # 关闭连接
        conn.close()

# 执行主函数
if __name__ == "__main__":
    print("正在加载IdeaBench数据库...")
    df = load_and_display_database()
    df.to_csv('./csvs/view.csv')
    
    if df is not None:
        # 可以添加更多分析代码
        print("\n数据库加载完成！可以继续进行更多分析...")
        
        # 如果需要交互式分析，可以保留df变量
        # 例如，你可以取消下面的注释以启用交互式分析
        # import code
        # code.interact(local=locals())
