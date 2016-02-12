import Quandl as q
rural = q.get('WORLDBANK/USA_SP_RUR_TOTL_ZS')
urban = q.get('WORLDBANK/USA_SP_URB_TOTL_IN_ZS')
# print(rural.to_dict())

import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)
plt.plot(rural.index,rural)
plt.xticks(rural.index[0::3],[])
plt.title('American Population')
plt.ylabel('% Rural')
plt.subplot(2, 1, 2)
plt.plot(urban.index,urban)
plt.xlabel('year')
plt.ylabel('% Urban')
plt.show()