import numpy as np

def remove_random_elements_except_first_last(my_array, num_elements_to_remove):
    # İlk ve son elemanları hariç tut
    my_array_without_ends = my_array[1:-1]
    
    # Silmek istediğiniz elemanların indekslerini rastgele seçin
    indices_to_remove = np.random.choice(len(my_array_without_ends), num_elements_to_remove, replace=False)
    
    # Seçilen indekslerdeki elemanları sil
    print('indices_to_remove: '+ str(indices_to_remove+1))
    modified_array = np.delete(my_array, indices_to_remove + 1)

    return modified_array

# Örnek bir dizi oluştur
my_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# İlk eleman ve son eleman hariç, rastgele elemanları sil
modified_array = remove_random_elements_except_first_last(my_array, 7)

print("Orijinal dizi:", my_array)
print("Değiştirilmiş dizi:", modified_array)
