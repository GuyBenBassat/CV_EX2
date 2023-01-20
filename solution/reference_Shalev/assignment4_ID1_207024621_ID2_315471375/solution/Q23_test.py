from utils import get_nof_params, load_model
from models import get_xception_based_model



model = get_xception_based_model()   #get the modified Xception network
print(get_nof_params(model))
