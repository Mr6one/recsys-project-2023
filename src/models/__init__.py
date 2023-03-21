from src.models.ngcf import *
from src.models.als import *
from src.models.eals import *
from src.models.ials import *

models = {
    'ngcf': NGCF,
    'als': ALS,
    'eals': eALS,
    'ials': iALS
}
