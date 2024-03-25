import numpy as np
import pytest
import matrices_multi as mm
import cProfile
import pstats
import logging
from memory_profiler import profile

"""
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
A_np = np.array(A)
B_np = np.array(B)
C = mm.matrices_multi(A, B)
print(C)
print(np.array(C))
print(np.dot(A_np, B_np))
"""

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='logs/pytest-logs.log',
                    filemode='w')
def pytest_configure(config):
    # Configure basic logging to a file with desired settings
    logging.basicConfig(filename="logs/pytest-logs.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#logging.basicConfig(level = logging.NOTSET, filename='test_logging.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.info('Test Begin')
log_cli = True
def test_matrix():
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[1, 2], [3, 4], [5, 6]]
    A_np = np.array(A)
    B_np = np.array(B)
    C = mm.matrices_multi(A, B)
    logging.info("This is an info message")
    assert np.array_equal(np.array(C), np.dot(A_np, B_np))



def test_square_matrix():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    A_np = np.array(A)
    B_np = np.array(B)
    C = mm.matrices_multi(A, B)
    print(C)
    assert np.array_equal(np.array(C), np.dot(A_np, B_np))

def test_vector_matrix():
    A = [[1, 2, 3]]
    B = [[1], [2], [3]]
    A_np = np.array(A)
    B_np = np.array(B)
    C = mm.matrices_multi(A, B)
    print(C)
    assert np.array_equal(np.array(C), np.dot(A_np, B_np))

def test_not_match_matrix():
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[1, 2], [3, 4], [5, 6], [7, 8]]
    with pytest.raises(ValueError):
        mm.matrices_multi(A, B)

def test_not_match_square_matrix():
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[1, 2], [3, 4]]
    with pytest.raises(ValueError):
        mm.matrices_multi(A, B)

def test_empty_matrix():
    A = []
    B = []
    A_np = np.array(A)
    B_np = np.array(B)
    C =  mm.matrices_multi(A, B)
    assert np.array_equal(C, np.dot(A_np, B_np))

def test_zero_matrix():
    A = B = [[0, 0], [0, 0]]
    C = mm.matrices_multi(A, B)
    A_np = np.array(A)
    B_np = np.array(B)
    C = mm.matrices_multi(A, B)
    assert np.array_equal(C, np.dot(A_np, B_np))

def test_large_matrix():
    A = np.random.rand(50, 100).tolist()
    B = np.random.rand(100, 50).tolist()
    C = mm.matrices_multi(A, B)
    A_np = np.array(A)
    B_np = np.array(B)
    C_exp = np.dot(A_np, B_np)
    for i in range(len(C)):
        for j in range(len(C[0])):
            assert abs(C[i][j] - C_exp[i][j]) < 1e-10 # Small differences due to floating-point precision

def test_complex_matrix():
    A = [[1+2j, 3+4j], [5+6j, 7+8j]]
    B = [[9+10j, 11+12j], [13+14j, 15+16j]]
    A_np = np.array(A)
    B_np = np.array(B)
    C = mm.matrices_multi(A, B)
    assert np.array_equal(np.array(C), np.dot(A_np, B_np))



def pytest_report_teststatus(report):
    # This hook is called for every test, and you can customize the logging here
    if report.when == 'call' or report.skipped:
        status = "PASSED" if report.passed else "FAILED" if report.failed else "SKIPPED"
        logging.info(f"{report.nodeid} - {status}")

logging.info('Test Finish')

#pylogging.basicConfig(level=logging.DEBUG, filename='test_log.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
# CPU profiling: pytest --profile matrics_multi_test.py
# Memory profiling: 
    # mprof run matrics_multi_test.py
    # mprof plot