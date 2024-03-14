# Wnioski ogólne 

## Rozkład wypadków
- Wyższa rozdzielczość (mniejsze heksagony) może uwydatnić lokalne obszary o dużej zmienności liczby wypadków, podczas gdy niższa rozdzielczość może pokazywać bardziej jednorodne obszary.
- Na niższych rozdzielczościach heksagony mają tendencję do grupowania większej liczby wypadków.
- Odchylenie standardowe liczby wypadków spada wraz ze wzrostem rozdzielczości. 

## Niezrównowazone cechy
- Największe niezrównowazone pod względem liczby cech w hexagonie dla klas występuje w grupie cech amenity.
- Istnieją cechy mocno niezrównowazone pod względem liczności dla klas, wykazujące wysoką korelację z wypadkami. Większość z nich nalezy do grupy amenity i highway.
- Wspólne niezrównowazone cechy dla resolution 8 i 9 (suma dla miast):
A_driver_t
A_langua_1
A_financia
A_coworkin
A_hunting_
H_trunk
R_loading_
H_raceway
A_bicycl_1
A_sailing_
H_bridlewa
A_smoking_
A_stables
A_boat_sto
A_stripclu
A_ferry_te
H_passing_
A_device_c
A_water_po
A_internet
A_weighbri
A_prison
A_animal_s
A_public_b
R_junction
A_motorcyc
A_receptio
H_track
A_loading_
R_traverse
H_motorway
H_motorw_1
R_owner_ch
R_containe
A_boat_ren
A_enclosin
R_site
A_cremator
R_spur_jun
A_universi
A_carpet_h
R_rail_bra
A_bakery
A_flight_s
A_office
A_bus_park
A_animal_1
R_wash
A_car_shar
A_post_dep
A_canteen
A_vacuum_c
A_ministry
A_watering
A_gambling
A_estate_a
H_pedestri
A_adult_ga
A_basin
A_accounti
R_hump_yar
A_dog_wash
RO_piste
H_emergenc
A_insuranc
A_police
R_razed
A_stage
A_securi_1
A_love_hot
A_studen_1
R_defect_d
A_gym

## Korelacja
Jako silna korelację przyjęto wartości powyzej 0.2 lub ponizej -0.2.

Cechy, które wykazują najsilniejszą korelację z wystąpieniem wypadków oraz powtarzające się dla resolution 8 i 9 (suma we wszystkich miastach Warszawa, Szczecin, Poznań)to:
H_service
H_tertiary
H_footway
H_resident
A_shelter
A_waste_ba
H_street_l
A_parking
H_platform
H_crossing
A_bicycle_
H_traffi_1
A_vending_
A_school
A_doctors
A_bicycl_1
A_cafe
H_bus_stop
A_restaura
A_dentist
A_pharmacy
A_atm
A_fast_foo


Te wspólne cechy mogą być istotne w kontekście analizy danych związanych z ruchem drogowym i wypadkami. Przypisanie większych wag tym cechom mogłoby polepszyć wyniki modelu.


- Udało się znaleźć cechy, które wykazują najsłabszą korelację z wystąpieniem wypadków oraz powtarzające się we wszystkich miastach (Warszawa, Szczecin, Poznań). Są one wypisane w notebookach.
- Dla największego resolution nie udało się znaleźć silnie skorelowanych cech.
- Im większe resolution tym mniejsza liczba istotnych cech.
- Wiele cech wykazuje silną pozytywną korelację pomiędzy sobą (mozemy zmniejszyć wymiarowość nie tracąc przy tym wiele informacji).
- Na podstawie przeprowadzonej analizy mozemy pominąc wiele cech. 
- Wyniki metryk dla korelacji cech z występowaniem wypadków wskazują na mniejszy wpływ lokalnych wzorców przestrzennych na występowanie wypadków drogowych w wyższych rozdzielczościach (średnia korelacja maleje wraz ze wzrostem resolution). 
- Dla zadnego miasta nie udało się znaleźć cech, które wykazują silną korelację z wystąpieniem wypadków i ich średnia występowania w hexagonie jest mniejsza niz 0.01.
- Istnieją cechy wykazujące wysoką korelację z wypadkami oraz występujące średnio przynajmniej raz w heksagonie. Większość z nich nalezy do grupy amenity i highway. 

## SHAP
- Średnia wartość Shapley'a (SHAP) maleje wraz ze wzrostem rozdzielczości dla Poznania i Szczecina. Warszawa osiągnęła najwyzsze wyniki (0.6 dla H_footway).
- Osiągnięte wartości sugerują, że wybrane cechy mają umiarkowany wpływ na przewidywania modelu. 
- Najwyzsze wyniki zostały osiągnięte dla:
H_footway
H_teritiary
H_service
H_resident
H_path
H_crossing
A_parking
A_bench

Interpretacja długości słupków: im dłuższy słupek dla danej klasy, tym większy jest wpływ tej zmiennej na prognozy dla tej klasy. Przykładowo, jeśli długość słupka dla klasy 1 jest znacząco większa niż dla klasy 0 dla danej zmiennej, oznacza to, że ta zmienna ma większy wpływ na prognozy dla klasy 1 niż dla klasy 0. Natomiast jeśli długości słupków dla obu klas są zbliżone, może to oznaczać, że ta zmienna ma podobny wpływ na obie klasy.
 

## Statystyka lokalna I Morana
Wskaźnik lokalny Moran'a (Local Moran's I) to narzędzie analizy przestrzennej, które pomaga zidentyfikować obszary o lokalnych wzorcach przestrzennej korelacji. Wskaźnik ten ocenia, czy wartości w określonych obszarach są skorelowane z wartościami w ich sąsiedztwie. Oznacza to stopień lokalnej korelacji przestrzennej.
p_sim to p-wartość uzyskana w wyniku testu statystycznego, który ocenia istotność statystyczną współczynnika Morana's I. P-wartość informuje o prawdopodobieństwie uzyskania danej wartości współczynnika Morana's I w próbie, gdyby hipoteza zerowa była prawdziwa (czyli brak przestrzennej korelacji). Niższa p-wartość oznacza większą istotność statystyczną, co prowadzi do odrzucenia hipotezy zerowej i sugeruje obecność przestrzennej korelacji.
Interpretacja Q:
- Q1 (High-High): Obszary o wysokich wartościach, które są otoczone przez obszary o wysokich wartościach.
- Q2 (Low-High): Obszary o niskich wartościach, otoczone przez obszary o wysokich wartościach.
- Q3 (Low-Low): Obszary o niskich wartościach, otoczone przez obszary o niskich wartościach. 
- Q4 (High-Low): Obszary o wysokich wartościach, otoczone przez obszary o niskich wartościach. 

#### Możliwe przyczyny zależności wartości moran_i, p-sim i q od resolution:
- Wraz ze wzrostem rozdzielczości (zmniejszeniem heksagonów), może się zmieniać charakterystyka danych przestrzennych. Mniejsze heksagony mogą uwydatniać bardziej lokalne zjawiska, podczas gdy większe heksagony mogą przynosić bardziej ogólne wzorce przestrzenne.
- Heterogeniczność obszarów. Wartości Moran'a I są wrażliwe na heterogeniczność obszarów. Obszary o różnych cechach mogą wpływać na siłę korelacji przestrzennej.

#### Wnioski:
- Wartości minimalne i wartość średnia p-value sugerują, że dla większości obszarów lokalne wzorce korelacji są istotne.
- Średnie wartości Q dla wszystkich rozdzielczości wskazują na to, że istnieją klastry obszarów o podobnych wartościach w sąsiedztwie. Dla rozdzielczości 10 maksymalna wartość Q wynosi 3, co znaczy, że obszary o wyższych wartościach są bardziej izolowane lub mają bardziej zróżnicowane sąsiedztwo.
