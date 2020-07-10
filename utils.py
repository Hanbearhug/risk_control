from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
"""
风险模型相关工具函数
作者：韩方园
日期：20200710
主要有一些风险中常用的指标计算函数，包括IV、PSI等等，analysis函数可用于得到IV、PSI稳定性指标，各分箱的好坏客户占比等指标。
"""
def cut_by_clf(x: pd.Series, y: pd.Series, n: int, nan: float = -99.) -> list:
    '''
    利用决策树获得最优分箱的边界值列表
    '''
    boundary = []  # 待return的分箱边界值列表

    x = x.fillna(nan).values  # 填充缺失值
    y = y.values

    clf = DecisionTreeClassifier(criterion='gini',  # “信息熵”最小化准则划分
                                 max_leaf_nodes=n,  # 最大叶子节点数
                                 min_samples_leaf=0.05)  # 叶子节点样本数量最小占比

    clf.fit(x.reshape(-1, 1), y)  # 训练决策树

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold

    for i in range(n_nodes):
        if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值
            boundary.append(threshold[i])

    boundary.sort()

    min_x = -np.inf
    max_x = np.inf  # +0.1是为了考虑后续groupby操作时，能包含特征最大值的样本
    boundary = [min_x] + boundary + [max_x]

    return boundary

# 计算static
def calc_static(cut_data,col,cut_bins):
    cut_grouped = cut_data.groupby(pd.cut(x=cut_data[col],bins=cut_bins,right=True))
    static_data = pd.DataFrame()
    static_data['all_cnt'] = cut_grouped['label'].count()
    static_data['label_1'] = cut_grouped['label'].sum()
    static_data['label_0'] = static_data['all_cnt']-static_data['label_1']
    static_data['all_ratio'] =  static_data['all_cnt']/np.sum(static_data['all_cnt'])
    static_data['label_0_ratio'] =  static_data['label_0']/np.sum(static_data['label_0'])
    static_data['label_1_ratio'] =  static_data['label_1']/np.sum(static_data['label_1'])
    static_data['label_1_rate'] = static_data['label_1']/(static_data['label_0']+static_data['label_1'])
    static_data['woe'] = np.log(static_data['label_1_ratio']/static_data['label_0_ratio'])
    static_data['iv']=(static_data['label_0_ratio']-static_data['label_1_ratio'])*np.log(static_data['label_0_ratio']/static_data['label_1_ratio'])
    iv = np.sum(static_data['iv'])
    bad_rate = np.sum(static_data['label_1'])/np.sum(static_data['label_1']+static_data['label_0'])
    static_data['risk_cor'] = static_data['label_1_rate']/bad_rate
    static_data['column'] = col
    if 'prob' in cut_data.columns:
        static_data['pred_mean'] = cut_grouped['prob'].mean()
    static_data.reset_index(inplace=True)
    static_data.rename(columns={col:'bins'},inplace=True)
    static_data.reset_index(inplace=True)
    static_data = static_data.rename(columns={'all_cnt': '客户数',
                                              'label_1': '坏客户数',
                                              'label_0': '好客户数',
                                              'all_ratio': '客户占比',
                                              'label_1_ratio': '坏客户占比',
                                              'label_0_ratio': '好客户占比',
                                              'label_1_rate': '采样Bad#',
                                              'bins': '变量值'})
    static_data['变量名称'] = col
    return bad_rate, iv, static_data.loc[:, ['变量名称', '变量值', '客户数', '坏客户数', '好客户数',
                                             '客户占比', '坏客户占比', '好客户占比', '采样Bad#']]


def psi_for_continue_var(expected_array, actual_array, bins=10, bucket_type='bins', detail=False, save_file_path=None):
    '''
    ----------------------------------------------------------------------
    功能: 计算连续型变量的群体性稳定性指标（population stability index ,PSI）
    ----------------------------------------------------------------------
    :param expected_array: numpy array of original values，基准组
    :param actual_array: numpy array of new values, same size as expected，比较组
    :param bins: number of percentile ranges to bucket the values into，分箱数, 默认为10
    :param bucket_type: string, 分箱模式，'bins'为等距均分，'quantiles'为按等频分箱
    :param detail: bool, 取值为True时输出psi计算的完整表格, 否则只输出最终的psi值
    :param save_file_path: string, csv文件保存路径. 默认值=None. 只有当detail=Ture时才生效.
    ----------------------------------------------------------------------
    :return psi_value:
            当detail=False时, 类型float, 输出最终psi计算值;
            当detail=True时, 类型pd.DataFrame, 输出psi计算的完整表格。最终psi计算值 = list(psi_value['psi'])[-1]
    ----------------------------------------------------------------------
    示例：
    >>> psi_for_continue_var(expected_array=df['score'][:400],
                             actual_array=df['score'][401:],
                             bins=5, bucket_type='bins', detail=0)
    >>> 0.0059132756739701245
    ------------
    >>> psi_for_continue_var(expected_array=df['score'][:400],
                             actual_array=df['score'][401:],
                             bins=5, bucket_type='bins', detail=1)
    >>>
    	score_range	expecteds	expected(%)	actucalsactucal(%)ac - ex(%)ln(ac/ex)psi	max
    0	[0.021,0.2095]	120.0	30.00	152.0	31.02	1.02	0.033434	0.000341
    1	(0.2095,0.398]	117.0	29.25	140.0	28.57	-0.68	-0.023522	0.000159
    2	(0.398,0.5865]	81.0	20.25	94.0	19.18	-1.07	-0.054284	0.000577	<<<<<<<
    3	(0.5865,0.7751]	44.0	11.00	55.0	11.22	0.22	0.019801	0.000045
    4	(0.7751,0.9636]	38.0	9.50	48.0	9.80	0.30	0.031087	0.000091
    5	>>> summary	400.0	100.00	489.0	100.00	NaN	NaN	0.001214	<<< result
    ----------------------------------------------------------------------
    知识:
    公式： psi = sum(（实际占比-预期占比）* ln(实际占比/预期占比))
    一般认为psi小于0.1时候变量稳定性很高，0.1-0.25一般，大于0.25变量稳定性差，建议重做。
    相对于变量分布(EDD)而言, psi是一个宏观指标, 无法解释两个分布不一致的原因。但可以通过观察每个分箱的sub_psi来判断。
    ----------------------------------------------------------------------
    '''
    import math
    import numpy as np
    import pandas as pd

    expected_array = pd.Series(expected_array).dropna()
    actual_array = pd.Series(actual_array).dropna()
    if isinstance(list(expected_array)[0], str) or isinstance(list(actual_array)[0], str):
        raise Exception("输入数据expected_array只能是数值型, 不能为string类型")

    """step1: 确定分箱间隔"""

    def scale_range(input_array, scaled_min, scaled_max):
        '''
        ----------------------------------------------------------------------
        功能: 对input_array线性放缩至[scaled_min, scaled_max]
        ----------------------------------------------------------------------
        :param input_array: numpy array of original values, 需放缩的原始数列
        :param scaled_min: float, 放缩后的最小值
        :param scaled_min: float, 放缩后的最大值
        ----------------------------------------------------------------------
        :return input_array: numpy array of original values, 放缩后的数列
        ----------------------------------------------------------------------
        '''
        input_array += -np.min(input_array)  # 此时最小值放缩到0
        if scaled_max == scaled_min:
            raise Exception('放缩后的数列scaled_min = scaled_min, 值为{}, 请检查expected_array数值！'.format(scaled_max))
        scaled_slope = np.max(input_array) * 1.0 / (scaled_max - scaled_min)
        input_array /= scaled_slope
        input_array += scaled_min
        return input_array

    # 异常处理，所有取值都相同时, 说明该变量是常量, 返回999999
    if np.min(expected_array) == np.max(expected_array):
        return 999999

    breakpoints = np.arange(0, bins + 1) / (bins) * 100  # 等距分箱百分比
    if 'bins' == bucket_type:  # 等距分箱
        breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
    elif 'quantiles' == bucket_type:  # 等频分箱
        breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

    """step2: 统计区间内样本占比"""

    def generate_counts(arr, breakpoints):
        '''
        ----------------------------------------------------------------------
        功能: Generates counts for each bucket by using the bucket values
        ----------------------------------------------------------------------
        :param arr: ndarray of actual values
        :param breakpoints: list of bucket values
        ----------------------------------------------------------------------
        :return cnt_array: counts for elements in each bucket, length of breakpoints array minus one
        :return score_range_array: 分箱区间
        ----------------------------------------------------------------------
        '''

        def count_in_range(arr, low, high, start):
            '''
            ----------------------------------------------------------------------
            功能: 统计给定区间内的样本数(Counts elements in array between low and high values)
            ----------------------------------------------------------------------
            :param arr: ndarray of actual values
            :param low: float, 左边界
            :param high: float, 右边界
            :param start: bool, 取值为Ture时，区间闭合方式[low, high],否则为(low, high]
            ----------------------------------------------------------------------
            :return cnt_in_range: int, 给定区间内的样本数
            ----------------------------------------------------------------------
            '''
            if start:
                cnt_in_range = len(np.where(np.logical_and(arr >= low, arr <= high))[0])
            else:
                cnt_in_range = len(np.where(np.logical_and(arr > low, arr <= high))[0])
            return cnt_in_range

        cnt_array = np.zeros(len(breakpoints) - 1)
        score_range_array = [''] * (len(breakpoints) - 1)
        for i in range(1, len(breakpoints)):
            cnt_array[i - 1] = count_in_range(arr, breakpoints[i - 1], breakpoints[i], i == 1)
            if 1 == i:
                score_range_array[i - 1] = '[' + str(round(breakpoints[i - 1], 4)) + ',' + str(
                    round(breakpoints[i], 4)) + ']'
            else:
                score_range_array[i - 1] = '(' + str(round(breakpoints[i - 1], 4)) + ',' + str(
                    round(breakpoints[i], 4)) + ']'

        return (cnt_array, score_range_array)

    expected_cnt = generate_counts(expected_array, breakpoints)[0]
    expected_percents = expected_cnt / len(expected_array)
    actual_cnt = generate_counts(actual_array, breakpoints)[0]
    actual_percents = actual_cnt / len(actual_array)
    delta_percents = actual_percents - expected_percents
    score_range_array = generate_counts(expected_array, breakpoints)[1]

    """step3: 区间放缩"""

    def sub_psi(e_perc, a_perc):
        '''
        ----------------------------------------------------------------------
        功能: 计算单个分箱内的psi值。Calculate the actual PSI value from comparing the values.
             Update the actual value to a very small number if equal to zero
        ----------------------------------------------------------------------
        :param e_perc: float, 期望占比
        :param a_perc: float, 实际占比
        ----------------------------------------------------------------------
        :return value: float, 单个分箱内的psi值
        ----------------------------------------------------------------------
        '''
        if a_perc == 0:  # 实际占比
            a_perc = 0.001
        if e_perc == 0:  # 期望占比
            e_perc = 0.001
        value = (e_perc - a_perc) * np.log(e_perc * 1.0 / a_perc)
        return value

    """step4: 得到最终稳定性指标"""
    sub_psi_array = [sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents))]
    if detail:
        psi_value = pd.DataFrame()
        psi_value['score_range'] = score_range_array
        psi_value['expecteds'] = expected_cnt
        psi_value['expected(%)'] = expected_percents * 100
        psi_value['actucals'] = actual_cnt
        psi_value['actucal(%)'] = actual_percents * 100
        psi_value['ac - ex(%)'] = delta_percents * 100
        psi_value['actucal(%)'] = psi_value['actucal(%)'].apply(lambda x: round(x, 2))
        psi_value['ac - ex(%)'] = psi_value['ac - ex(%)'].apply(lambda x: round(x, 2))
        psi_value['ln(ac/ex)'] = psi_value.apply(lambda row: np.log((row['actucal(%)'] + 0.001) \
                                                                    / (row['expected(%)'] + 0.001)), axis=1)
        psi_value['psi'] = sub_psi_array
        flag = lambda x: '<<<<<<<' if x == psi_value.psi.max() else ''
        psi_value['max'] = psi_value.psi.apply(flag)
        psi_value = psi_value.append([{'score_range': '>>> summary',
                                       'expecteds': sum(expected_cnt),
                                       'expected(%)': 100,
                                       'actucals': sum(actual_cnt),
                                       'actucal(%)': 100,
                                       'ac - ex(%)': np.nan,
                                       'ln(ac/ex)': np.nan,
                                       'psi': np.sum(sub_psi_array),
                                       'max': '<<< result'}], ignore_index=True)
        if save_file_path:
            if not isinstance(save_file_path, str):
                raise Exception('参数save_file_path类型必须是str, 同时注意csv文件后缀!')
            elif not save_file_path.endswith('.csv'):
                raise Exception('参数save_file_path不是csv文件后缀，请检查!')
            psi_value.to_csv(save_file_path, encoding='utf-8', index=1)
    else:
        psi_value = np.sum(sub_psi_array)

    return psi_value

def analysis(data, cols, all_cut_bins=[], date_col='申请时间'):
    data['申请时间'] = pd.to_datetime(data['申请时间'])

    Rank_df = pd.DataFrame()
    IV_df = pd.DataFrame()
    PSI_df = pd.DataFrame()
    IV_all = pd.DataFrame()


    bins = 5
    nan_value = -999
    for i, col in enumerate(cols):
        # 取第一个月份作为base计算最优的分箱阈值
        data['month'] = data[date_col].dt.month
        data['year'] = data[date_col].dt.year
        start_year = data[date_col].min().year
        start_month = data[date_col].min().month

        data_base = data.loc[(data['year'] == start_year) &
                             (data['month'] == start_month)]

        if len(all_cut_bins)==0:
            cut_bins = cut_by_clf(data_base[col], data_base['label'], n=bins, nan=nan_value)
        else:
            cut_bins = all_cut_bins[i]
        # 计算分箱下的特征woe、iv等
        bad_rate, iv, rank_df = calc_static(data.fillna(nan_value), col, cut_bins)
        Rank_df = pd.concat([Rank_df, rank_df])

        iv_df = pd.DataFrame([[col, iv]], columns=['变量名称', 'iv'])
        IV_all = pd.concat([IV_all, iv_df])

        # IV、PSI稳定性情况，按照月份观察PSI和IV的变化情况，将第一个月作为基准月
        end_year = data[date_col].max().year
        end_month = data[date_col].max().month
        IV_time = []
        PSI_time = []
        time_cols = []

        for year in range(start_year, end_year+1):
            if year == start_year:
                month_s = start_month
            else:
                month_s = 1
            if year == end_year:
                month_e = end_month
            else:
                month_e = 12
            for month in range(month_s, month_e+1):
                time_col = str(year) + '/' + str(month)
                data_time = data.loc[(data['year'] == year) &
                                     (data['month'] == month)]
                bad_rate_time, iv_time, rank_df_time = calc_static(data_time.fillna(nan_value), col, cut_bins)
                psi_time = psi_for_continue_var(data_base[col], data_time[col])


                time_cols.append(time_col)
                IV_time.append(iv_time)
                PSI_time.append(psi_time)

        IV_col = pd.DataFrame([IV_time], columns=time_cols)
        PSI_col = pd.DataFrame([PSI_time], columns=time_cols)
        IV_col['变量名称'] = col
        PSI_col['变量名称'] = col

        IV_df = pd.concat([IV_df, IV_col])
        PSI_df = pd.concat([PSI_df, PSI_col])

    # 相关性情况
    corr_df = data[cols].corr()
    return Rank_df, IV_df, IV_all, PSI_df, corr_df







