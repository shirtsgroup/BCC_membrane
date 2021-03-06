#! /usr/bin/env python

import make_dict

# Dictionary of atomic masses associated with various atom types

mass = {}
charge = {}
radius = {}

mass['C'] = 12.01
mass['C1'] = 12.01
mass['C2'] = 12.01
mass['C3'] = 12.01
mass['C4'] = 12.01
mass['C5'] = 12.01
mass['C6'] = 12.01
mass['C7'] = 12.01
mass['C8'] = 12.01
mass['C9'] = 12.01
mass['C10'] = 12.01
mass['C11'] = 12.01
mass['C12'] = 12.01
mass['C13'] = 12.01
mass['C14'] = 12.01
mass['C15'] = 12.01
mass['C16'] = 12.01
mass['C17'] = 12.01
mass['C18'] = 12.01
mass['C19'] = 12.01
mass['C20'] = 12.01
mass['C21'] = 12.01
mass['C22'] = 12.01
mass['C23'] = 12.01
mass['C24'] = 12.01
mass['C25'] = 12.01
mass['C26'] = 12.01
mass['C27'] = 12.01
mass['C28'] = 12.01
mass['C29'] = 12.01
mass['C30'] = 12.01
mass['C31'] = 12.01
mass['C32'] = 12.01
mass['C33'] = 12.01
mass['C34'] = 12.01
mass['C35'] = 12.01
mass['C36'] = 12.01
mass['C37'] = 12.01
mass['C38'] = 12.01
mass['C39'] = 12.01
mass['C40'] = 12.01
mass['C41'] = 12.01
mass['C42'] = 12.01
mass['C43'] = 12.01
mass['C44'] = 12.01
mass['C45'] = 12.01
mass['C46'] = 12.01
mass['C47'] = 12.01
mass['C48'] = 12.01
mass['O'] = 16.00
mass['O1'] = 16.00
mass['O2'] = 16.00
mass['O3'] = 16.00
mass['O4'] = 16.00
mass['O5'] = 16.00
mass['O6'] = 16.00
mass['O7'] = 16.00
mass['O8'] = 16.00
mass['O9'] = 16.00
mass['O10'] = 16.00
mass['H'] = 1.008
mass['H1'] = 1.008
mass['H2'] = 1.008
mass['H3'] = 1.008
mass['H4'] = 1.008
mass['H5'] = 1.008
mass['H6'] = 1.008
mass['H7'] = 1.008
mass['H8'] = 1.008
mass['H9'] = 1.008
mass['H10'] = 1.008
mass['H11'] = 1.008
mass['H12'] = 1.008
mass['H13'] = 1.008
mass['H14'] = 1.008
mass['H15'] = 1.008
mass['H16'] = 1.008
mass['H17'] = 1.008
mass['H18'] = 1.008
mass['H19'] = 1.008
mass['H20'] = 1.008
mass['H21'] = 1.008
mass['H22'] = 1.008
mass['H23'] = 1.008
mass['H24'] = 1.008
mass['H25'] = 1.008
mass['H26'] = 1.008
mass['H27'] = 1.008
mass['H28'] = 1.008
mass['H29'] = 1.008
mass['H30'] = 1.008
mass['H31'] = 1.008
mass['H32'] = 1.008
mass['H33'] = 1.008
mass['H34'] = 1.008
mass['H35'] = 1.008
mass['H36'] = 1.008
mass['H37'] = 1.008
mass['H38'] = 1.008
mass['H39'] = 1.008
mass['H40'] = 1.008
mass['H41'] = 1.008
mass['H42'] = 1.008
mass['H43'] = 1.008
mass['H44'] = 1.008
mass['H45'] = 1.008
mass['H46'] = 1.008
mass['H47'] = 1.008
mass['H48'] = 1.008
mass['H49'] = 1.008
mass['H50'] = 1.008
mass['H51'] = 1.008
mass['H52'] = 1.008
mass['H53'] = 1.008
mass['H54'] = 1.008
mass['H55'] = 1.008
mass['H56'] = 1.008
mass['H57'] = 1.008
mass['H58'] = 1.008
mass['H59'] = 1.008
mass['H60'] = 1.008
mass['H61'] = 1.008
mass['H62'] = 1.008
mass['H63'] = 1.008
mass['H64'] = 1.008
mass['H65'] = 1.008
mass['H66'] = 1.008
mass['H67'] = 1.008
mass['H68'] = 1.008
mass['H69'] = 1.008
mass['H70'] = 1.008
mass['H71'] = 1.008
mass['H72'] = 1.008
mass['H73'] = 1.008
mass['H74'] = 1.008
mass['H75'] = 1.008
mass['H76'] = 1.008
mass['H77'] = 1.008
mass['H78'] = 1.008
mass['H79'] = 1.008
mass['H80'] = 1.008
mass['H81'] = 1.008
mass['H82'] = 1.008
mass['H83'] = 1.008
mass['NA'] = 22.98977
mass['CL'] = 35.453
mass['OW'] = 15.99940
mass['HW1'] = 1.008
mass['HW2'] = 1.008
mass['HW'] = 1.008
mass['BR'] = 79.90
mass['N'] = 14.01
mass['N1'] = 14.01
mass['N2'] = 14.01
mass['N3'] = 14.01
mass['OW_tip4pew'] = 16.00000
mass['HW_tip4pew'] = 1.008
mass['HW_tip4pew'] = 1.008
mass['MW'] = 0.0
mass['OXT'] = 15.99940
mass['CA'] = 40.08
mass['S'] = 32.06
mass['S1'] = 32.06

# fill in charge dictionary (Used make_dict.py to generate the following list)


def charges(monomer):

    c = make_dict.mk_dict(monomer)

    return c

charge['C'] = -0.0946
charge['C1'] = -0.1495
charge['C2'] = 0.0776
charge['C3'] = 0.0151
charge['C4'] = 0.0776
charge['C5'] = -0.1495
charge['O'] = -0.3429
charge['O1'] = -0.3389
charge['O2'] = -0.3429
charge['C6'] = 0.902202
charge['O3'] = -0.824301
charge['O4'] = -0.824301
charge['C7'] = 0.1299
charge['C8'] = -0.1004
charge['C9'] = -0.0759
charge['C10'] = -0.0804
charge['C11'] = -0.0799
charge['C12'] = -0.0799
charge['C13'] = -0.0789
charge['C14'] = -0.0779
charge['C15'] = -0.0799
charge['C16'] = -0.0979
charge['C17'] = 0.1354
charge['O5'] = -0.4384
charge['C18'] = 0.631301
charge['C19'] = -0.2082
charge['C20'] = -0.139
charge['C21'] = 0.1274
charge['C22'] = -0.0784
charge['C23'] = -0.0814
charge['C24'] = -0.0774
charge['C25'] = -0.0814
charge['C26'] = -0.0784
charge['C27'] = -0.0814
charge['C28'] = -0.0774
charge['C29'] = -0.0824
charge['C30'] = -0.1074
charge['C31'] = 0.1354
charge['O6'] = -0.4439
charge['C32'] = 0.633301
charge['C33'] = -0.2142
charge['C34'] = -0.13
charge['C35'] = 0.1299
charge['C36'] = -0.1004
charge['C37'] = -0.0759
charge['C38'] = -0.0804
charge['C39'] = -0.0799
charge['C40'] = -0.0799
charge['C41'] = -0.0789
charge['C42'] = -0.0779
charge['C43'] = -0.0799
charge['C44'] = -0.0979
charge['C45'] = 0.1354
charge['O7'] = -0.4384
charge['C46'] = 0.631301
charge['C47'] = -0.2082
charge['C48'] = -0.139
charge['O8'] = -0.559501
charge['O9'] = -0.546001
charge['O10'] = -0.559501
charge['H'] = 0.1665
charge['H1'] = 0.1665
charge['H2'] = 0.0447
charge['H3'] = 0.0447
charge['H4'] = 0.05395
charge['H5'] = 0.05395
charge['H6'] = 0.03945
charge['H7'] = 0.03945
charge['H8'] = 0.03895
charge['H9'] = 0.03895
charge['H10'] = 0.0437
charge['H11'] = 0.0437
charge['H12'] = 0.04095
charge['H13'] = 0.04095
charge['H14'] = 0.0367
charge['H15'] = 0.0367
charge['H16'] = 0.0407
charge['H17'] = 0.0407
charge['H18'] = 0.0452
charge['H19'] = 0.0452
charge['H20'] = 0.0532
charge['H21'] = 0.0532
charge['H22'] = 0.0622
charge['H23'] = 0.0622
charge['H24'] = 0.1685
charge['H25'] = 0.12675
charge['H26'] = 0.12675
charge['H27'] = 0.0417
charge['H28'] = 0.0417
charge['H29'] = 0.0432
charge['H30'] = 0.0432
charge['H31'] = 0.0482
charge['H32'] = 0.0482
charge['H33'] = 0.0357
charge['H34'] = 0.0357
charge['H35'] = 0.0407
charge['H36'] = 0.0407
charge['H37'] = 0.0377
charge['H38'] = 0.0377
charge['H39'] = 0.0397
charge['H40'] = 0.0397
charge['H41'] = 0.0387
charge['H42'] = 0.0387
charge['H43'] = 0.0472
charge['H44'] = 0.0472
charge['H45'] = 0.0572
charge['H46'] = 0.0572
charge['H47'] = 0.0652
charge['H48'] = 0.0652
charge['H49'] = 0.152
charge['H50'] = 0.1295
charge['H51'] = 0.1295
charge['H52'] = 0.0447
charge['H53'] = 0.0447
charge['H54'] = 0.05395
charge['H55'] = 0.05395
charge['H56'] = 0.03945
charge['H57'] = 0.03945
charge['H58'] = 0.03895
charge['H59'] = 0.03895
charge['H60'] = 0.0437
charge['H61'] = 0.0437
charge['H62'] = 0.04095
charge['H63'] = 0.04095
charge['H64'] = 0.0367
charge['H65'] = 0.0367
charge['H66'] = 0.0407
charge['H67'] = 0.0407
charge['H68'] = 0.0452
charge['H69'] = 0.0452
charge['H70'] = 0.0532
charge['H71'] = 0.0532
charge['H72'] = 0.0622
charge['H73'] = 0.0622
charge['H74'] = 0.1685
charge['H75'] = 0.12675
charge['H76'] = 0.12675
charge['H77'] = 0.00000
charge['H78'] = 0.00000
charge['H79'] = 0.00000
charge['H80'] = 0.00000
charge['H81'] = 0.00000
charge['H82'] = 0.00000
charge['H83'] = 0.00000
charge['NA'] = 1.00000
charge['CL'] = -1.00000
charge['OW'] = -0.834
charge['HW1'] = .417
charge['HW2'] = .417
charge['BR'] = -1.00000
charge['CA'] = 2.00000

radius['C'] = 170  # angstroms
radius['O'] = 152
radius['H'] = 120
radius['NA'] = 227
radius['Na'] = 227
