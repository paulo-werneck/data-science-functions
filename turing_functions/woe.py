def TuringClassInformationValueWoEMetrics(data_frame: '<class pandas.core.frame.DataFrame>',
                                          target: str,
                                          variables: list,
                                          missing: str = 'missing') -> ('<class pandas.core.frame.DataFrame>', list):
    """
    :param data_frame:  initial data frame
    :param target:      target variable contained in the data_frame parameter
    :param variables:   list of variables contained in the data_frame parameter
    :param missing:     Value to be inserted in categories with no values
    :return[0]:         data frame of Weight of Evidence (WoE)
    :return[1]:         list of tuples of Information Value (IV)
    """

    from operator import countOf
    import numpy as np
    import pandas as pd

    if len(data_frame[target].unique()) > 2:
        raise Exception('Your target is not binary, please check')

    qtd_all_targets = [countOf(data_frame[target], x) for x in range(0, 2)]
    df_woe = pd.DataFrame(columns=['Variavel', 'Categoria', 'WoE'])
    lst_iv = list()
    index = 0

    for variable in variables:
        data_frame[variable] = [missing if pd.isna(x) else x for x in data_frame[variable]]
        iv = 0

        for category in data_frame[variable].unique():
            df_aux = data_frame[data_frame[variable] == category]

            try:
                goods, bads = [(countOf(df_aux[target], x)) / qtd_all_targets[x] for x in range(0, 2)]
                woe = np.log(goods / bads)
            except ZeroDivisionError:
                print('division by zero occurred, please check your target')
            else:
                df_woe.loc[index] = [variable, category, woe]
                iv = iv + woe * (goods - bads)
            index += 1
        lst_iv.append((variable, round(iv, 10)))

    df_woe.sort_values(by=['Variavel', 'WoE'], ascending=True, inplace=True)
    df_woe = df_woe.reset_index(drop=True)

    return df_woe, lst_iv
