import random

def tahmin_oyunu():
    secret = random.randint(1, 10)
    print("1 ile 10 arasında rastgele seçilen sayıyı tahmin edin. Toplam 3 hakkınız var.")

    for hak in range(1, 4):
        try:
            tahmin = int(input(f"{hak}. tahmininizi girin: "))
        except Exception as e:
            print("Hata: Lütfen geçerli bir sayı girin.", e)
            continue  # Hatalı girişte, hakkı kaybetmeden devam edebiliriz

        if tahmin == secret:
            print("Tebrikler! Doğru tahmin ettiniz!")
            return
        elif tahmin < secret:
            print("Daha yüksek bir sayı deneyin.")
        else:
            print("Daha düşük bir sayı deneyin.")
    
    print("Üzgünüm, hakkınız bitti. Doğru sayı:", secret)

tahmin_oyunu()
