# Subtask 1.4
import unittest
import pandas as pd
from Classes import DAO
from Classes import SubTask13


# as a reference!
#  def __init__(self, ifname, ofname, iftype, oftype):
class UnitTests(unittest.TestCase):

    def test_type_is_type(self):
        # The file and the type of file do not correspond
        with self.assertRaises(Exception) as context:
            dao = DAO('test_1.txt', 'test_1_out.txt', '.csv', '.txt')
        self.assertTrue("The file and the type of file do not correspond!")

    def test_reading(self):
        # check the reading of the .csv file
        dao = DAO('Test_2.txt', 'Test_2_output.txt', '.txt', '.txt')
        data = dao.load()
        dataframe = pd.DataFrame([[2, 4, 7], [7, 1, 8], [6, 0, 2]])
        self.assertTrue(dataframe.equals(data))

    def test_writing(self):
        # check the writing to the memory
        dao = DAO('Test_2.txt', 'Test_2_output.txt', '.txt', '.txt')
        data = dao.load()
        # store returns a Boolean
        temp = dao.store(data)
        self.assertTrue(temp)

    def test_process1(self):
        # check if SubTask13 successfully received DAO object
        dao = DAO('input13.txt', 'output13.txt', '.txt', '.txt')
        subTask13 = SubTask13(dao)
        self.assertEqual(subTask13.my_dao, dao)

    def test_process2(self):
        # check if SubTask13 successfully processed 'abcdefghi'
        dao = DAO('input13.txt', 'output13.txt', '.txt', '.txt')
        sub_task = SubTask13(dao)
        sub_task.process()

        with open(sub_task.my_dao.ofname, 'r') as file:
            f = file.read()
        file.close()
        # I saved the file with \n at the end!!
        self.assertEqual(f, 'bdfhigeca\n')

    def test_process3(self):
        # check if SubTask13 successfully processed 'Starcraft'
        dao = DAO('test_3.txt', 'test_3_output.txt', '.txt', '.txt')
        sub_task = SubTask13(dao)
        sub_task.process()

        with open(sub_task.my_dao.ofname, 'r') as file:
            f = file.read()
        file.close()
        # I printed a \n at the end of the line
        self.assertEqual(f, 'trrftacaS\n')


if __name__ == 'main':
    unittest.main()
