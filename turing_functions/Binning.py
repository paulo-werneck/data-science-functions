def TuringOptimalBinning(data_frame, target, *args, **kwargs):
    """
    -- Esta funcao categoriza variaveis continuas usando arvore de decisao.
    -- A quantidade de categorias geradas devera ser igual ou aproximada da quantidade desejada

    :param data_frame:  Data Frame
    :param target:      Variavel Target
    :param args:        N Tuplas com pares de variaveis e numero de categorias desejadas
    :param kwargs:      N pares de variaveis e numero de categorias desejadas
    :return[0]:         Data Frame com as variaveis categorizada
    :return[1]:         Data Frame com o intervalo de dados das categorias

    sample: TuringOptimalBinning(data_frame, 'target', ([var1, var2, var3, varn], nr_bins), var1=nr_bins, var2=nr_bins, varn=nr_bins)
    """

    import xgboost as xgb
    import pandas as pd
    import numpy as np

    data_frame2 = pd.DataFrame(columns=['Var', 'Bin', 'Count', 'Qtd_evento', 'Tx_evento', 'Min', 'Max'])
    create_xgb = None

    if args is not None:
        for v in args:
            dct = dict.fromkeys(v[0], v[1])
            kwargs.update(dct)

    for c, v in kwargs.items():

        # separacao dos data_frames
        df_complete = data_frame.loc[:, [c, target]]
        df_not_missing = df_complete[df_complete[c].isna() == False]
        df_missing = df_complete[df_complete[c].isna() == True]

        name_col_bin = 'bin_' + c

        for j in range(v, 0, -1):
            try:
                create_xgb = xgb.XGBClassifier(n_estimators=j, max_depth=2)
                create_xgb.fit(df_not_missing[c].to_frame(), df_not_missing[target])
            except ValueError:
                print("Favor informar apenas variáveis discretas e ou contínuas para categorização.")

            lst_predict = create_xgb.predict_proba(df_not_missing[c].to_frame())[:, 1]
            qtd_bins_curr = len(np.unique(lst_predict))

            # para o laco no melhor estimador
            if qtd_bins_curr <= v:
                break

        # juncao dos data_frames
        df_not_missing.insert(1, name_col_bin, lst_predict, True)
        df_final = df_not_missing.append(df_missing, ignore_index=False).sort_index()
        df_final[df_final[name_col_bin].isna() == True] = -99999
        data_frame = data_frame.join(df_final[name_col_bin])

        # renomeia os bins
        for i, j in enumerate(data_frame[name_col_bin].unique()):
            data_frame[name_col_bin].replace(to_replace=j, value='B' + str(i), inplace=True)

        # gera data_frame das regras
        agg = pd.concat([
            data_frame.groupby([name_col_bin])[target].count(),
            data_frame.groupby([name_col_bin])[target].sum(),
            round(data_frame.groupby([name_col_bin])[target].mean() * 100, 2),
            round(data_frame.groupby([name_col_bin])[c].min(), 10),
            round(data_frame.groupby([name_col_bin])[c].max(), 10)
        ], axis=1).reset_index()
        agg.columns = ['Bin', 'Count', 'Qtd_evento', 'Tx_evento', 'Min', 'Max']
        agg['Var'] = name_col_bin
        data_frame2 = data_frame2.append(agg, ignore_index=True)
        data_frame2.sort_values(['Var', 'Tx_evento', 'Min'], ascending=[True, False, True], inplace=True)

    return data_frame, data_frame2


def TuringCategoryBinning(data_frame, data_frame_rules, *args):
    """
    -- Funcao para categorizacao de variaveis (Binning)
    -- As regras utilizadas para categorizacao sao baseadas nas que foram geradas pela funcao TuringOptimalBinning

    :param data_frame:          Data Frame
    :param data_frame_rules:    Data Frame com as regras de categorizacao
    :param args:                Lista ou tupla de variaveis a serem categorizadas
    :return:                    Data Frame de entrada + variaveis categorizadas

    sample: TuringCategoryBinning(data_frame, data_frame_rules, *('var1', 'var2', 'var3', 'varN'))
    """

    import pandas as pd
    import numpy as np
    import math

    col_bin_name = None
    bin_null = None

    for i in args:
        col_bin_name = 'bin_' + i
        df_var = data_frame[i].to_frame()
        df_var.columns = [col_bin_name]

        df_aux = pd.DataFrame(columns=[col_bin_name])
        df_aux2 = pd.DataFrame(columns=[col_bin_name])

        a = data_frame_rules[data_frame_rules['Var'] == col_bin_name].to_dict()
        vbin, vmin, vmax = a['Bin'], a['Min'], a['Max']

        for j in sorted(vbin.keys()):
            df_aux[col_bin_name] = df_var[col_bin_name].apply(lambda x: vbin[j] if vmin[j] <= x <= vmax[j] else 0)
            df_aux2 = df_aux2.append(df_aux[df_aux[col_bin_name] != 0])

            if math.isnan(vmin[j]):
                bin_null = vbin[j]

        data_frame = data_frame.join(df_aux2)
        data_frame[col_bin_name].replace(to_replace=np.nan, value=bin_null, inplace=True)

    print(f"Quantitativo de itens por categorizações")
    print(data_frame.groupby([col_bin_name])[i].count().to_frame().sort_values(i, ascending=False))

    return data_frame
