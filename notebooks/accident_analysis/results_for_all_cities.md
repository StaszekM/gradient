# Wnioski ogólne 

## Rozkład wypadków
- Wyższa rozdzielczość (mniejsze heksagony) może uwydatnić lokalne obszary o dużej zmienności liczby wypadków, podczas gdy niższa rozdzielczość może pokazywać bardziej jednorodne obszary.

## Korelacja i SHAP

## Warszawa
Common Features Across All Resolutions:
{'highway_fo', 'highway_re', 'highway__9', 'railway_16', 'highway__5', 'highway_bu', 'highway_11', 'railway_18', 'highway_cy', 'highway_st', 'highway_se', 'highway_cr', 'highway_14', 'highway_te', 'highway_tr', 'railway_17'}

## Szczecin
Common Features Across All Resolutions:
{'highway_st', 'highway_re', 'highway_13', 'highway__9', 'highway_bu', 'highway_fo', 'highway_te', 'highway_cr'}

## Poznań
Common Features Across All Resolutions:
{'highway_te', 'highway_18', 'highway_fo', 'highway_pl', 'highway_bu', 'highway_cr', 'highway_15', 'highway_11'}


Cechy, które powtarzają się we wszystkich miastach (Warszawa, Szczecin, Poznań) to:
- 'highway_fo' - footway
- 'highway_te' - tertiary (The next most important roads in a country's system. (Often link smaller towns and villages))
- 'highway__cr' - crossing
- 'highway_bu' - bus_stop

Te wspólne cechy mogą być istotne w kontekście analizy danych związanych z ruchem drogowym i wypadkami. Przypisanie większych wag tym cechom mogłoby polepszyć wyniki modelu.

#### Wnioski 
- Im większe resolution tym większa liczba względnie istotnych cech.
 - Średnia wartość Shapley'a (SHAP) maleje wraz ze wzrostem rozdzielczości. Osiągnięte wartości sugerują, że wybrane cechy mają niewielki wpływ na przewidywania modelu. 
 - Wiele cech wykazuje silną pozytywną korelację pomiędzy sobą (mozemy zmniejszyć wymiarowość nie tracąc duzo informacji).
 - Wyniki metryk dla korelacji cech z występowaniem wypadków wskazują na mniejszy wpływ lokalnych wzorców przestrzennych na występowanie wypadków drogowych w wyższych rozdzielczościach. 
 - Na podstawie przeprowadzonej analizy mozemy pominąc wiele cech. 

## Statystyka lokalna I Morana
Wskaźnik lokalny Moran'a (Local Moran's I) to narzędzie analizy przestrzennej, które pomaga zidentyfikować obszary o lokalnych wzorcach przestrzennej korelacji. Wskaźnik ten ocenia, czy wartości w określonych obszarach są skorelowane z wartościami w ich sąsiedztwie. Oznacza to stopień lokalnej korelacji przestrzennej.
p_sim to p-wartość uzyskana w wyniku testu statystycznego, który ocenia istotność statystyczną współczynnika Morana's I. P-wartość informuje o prawdopodobieństwie uzyskania danej wartości współczynnika Morana's I w próbie, gdyby hipoteza zerowa była prawdziwa (czyli brak przestrzennej korelacji). Niższa p-wartość oznacza większą istotność statystyczną, co prowadzi do odrzucenia hipotezy zerowej i sugeruje obecność przestrzennej korelacji.
Interpretacja Q:
- Q1 (High-High): Obszary o wysokich wartościach, które są otoczone przez obszary o wysokich wartościach.
- Q2 (Low-High): Obszary o niskich wartościach, otoczone przez obszary o wysokich wartościach.
- Q3 (Low-Low): Obszary o niskich wartościach, otoczone przez obszary o niskich wartościach. 
- Q4 (High-Low): Obszary o wysokich wartościach, otoczone przez obszary o niskich wartościach. 

Możliwe przyczyny zależności wartości moran_i, p-sim i q od resolution:
- Wraz ze wzrostem rozdzielczości (zmniejszeniem heksagonów), może się zmieniać charakterystyka danych przestrzennych. Mniejsze heksagony mogą uwydatniać bardziej lokalne zjawiska, podczas gdy większe heksagony mogą przynosić bardziej ogólne wzorce przestrzenne.
- Heterogeniczność obszarów. Wartości Moran'a I są wrażliwe na heterogeniczność obszarów. Obszary o różnych cechach mogą wpływać na siłę korelacji przestrzennej.

### Wnioski ogólne:
- Wartości minimalne i wartość średnia p-value sugerują, że dla większości obszarów lokalne wzorce korelacji są istotne.
- Średnie wartości Q dla wszystkich rozdzielczości wskazują na to, że istnieją klastry obszarów o podobnych wartościach w sąsiedztwie. Dla rozdzielczości 10 maksymalna wartość Q wynosi 3, co znaczy, że obszary o wyższych wartościach są bardziej izolowane lub mają bardziej zróżnicowane sąsiedztwo.
