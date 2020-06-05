from turing_functions.woe import TuringClassInformationValueWoEMetrics
from pandas.testing import assert_frame_equal
import pandas as pd
from unittest import TestCase
from parameterized import parameterized
import os


class TestWoe(TestCase):

    def setUp(self):
        print(os.system('ls'))
        pwd = os.environ['PWD'].split('/')[:-1]
        pwd.append('utils')
        path = '/'.join(pwd)
        df = pd.read_csv(path+'/df_input_test.csv')
        self.woe, self.iv = TuringClassInformationValueWoEMetrics(df, 'target', ['blood_group', 'sex'])

    def test_qtd_variables_woe(self):
        qtd_var_woe = len(self.woe['Variavel'].unique())
        self.assertEqual(qtd_var_woe, 2)

    def test_qtd_variables_iv(self):
        self.assertEqual(len(self.iv), 2)

    def test_verify_calc_woe(self):
        df_woe = pd.DataFrame(data=[
            ('blood_group', 'A-', 0.1270619635),
            ('blood_group', 'A+', 0.1323003202),
            ('blood_group', 'AB-', 0.2895808930),
            ('blood_group', 'AB+', 0.0325357901),
            ('blood_group', 'B-', -0.1190141081),
            ('blood_group', 'B+', -0.2471141107),
            ('blood_group', 'O-', 0.0733577846),
            ('blood_group', 'O+', -0.3633598670),
            ('sex', 'F', -0.0553500951),
            ('sex', 'M', 0.0585941643)],
            columns=['Variavel', 'Categoria', 'WoE'])
        df_woe.sort_values(by=['Variavel', 'WoE'], ascending=True, inplace=True)
        df_woe = df_woe.reset_index(drop=True)

        assert_frame_equal(df_woe, self.woe)

    @parameterized.expand([
        ['blood_group', 0.0422449200],
        ['sex', 0.0032423163]
    ])
    def test_verify_calc_iv(self, name, p_iv):
        iv = [i for _, i in self.iv]
        self.assertIn(p_iv, iv)
