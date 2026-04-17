# TestEm13 için Geliştirilmiş (Daha Gerçekçi) Cs-137 Kurulumu

Bu klasör, verdiğin fotoğraf ve ölçülere göre TestEm13'e doğrudan taşınabilecek **geliştirilmiş geometri + kaynak üretim** dosyalarını içerir.

## 1) Bu pakette hangi dosyalar var?

- `DetectorConstruction.hh/.cc`
  - Metal kabin (Cu kabuk + iç hava)
  - Polikarbonat kaynak kapsülü + merkez aktif bölge
  - Pb kolimatör (9x9x9 cm, delik çapı 2 mm)
  - Numune bölgesi
  - HPGe için katmanlı model (Al endcap + vakum + Be pencere + Ge dead layer + aktif Ge)
  - Alternatif NaI(Tl)
- `PrimaryGeneratorAction.hh/.cc`
  - Cs-137 için **sabit enerji yerine gaussian enerji dağılımı**
  - Ortalama: 661.657 keV
  - Sigma: 0.80 keV (pratik ölçüm yayılımı temsili)
- `ActionInitialization.hh/.cc`
  - Primary generator bağlantısı
- `run_cs137_gauss.mac`
  - Hızlı görselleştirme + örnek beamOn

## 2) Geometri parametreleri (senin verdiğin değerlere göre)

- Doğrusal dizilim: **Kaynak → Kolimatör → Numune → Dedektör**
- Kaynak–dedektör uzaklığı: **38 cm** (kaynak merkezi ile aktif Ge kristal merkezi arası)
- Kolimatör: **Pb küp 9×9×9 cm**, delik çapı **2 mm**
- Kaynak: polikarbonat kapsül, aktif nokta merkezde

## 3) TestEm13'e entegrasyon

Geant4 örnek ağacında:

`examples/extended/electromagnetic/TestEm13`

içine bu dosyaları kopyala/değiştir:

- `DetectorConstruction.hh`
- `DetectorConstruction.cc`
- `PrimaryGeneratorAction.hh`
- `PrimaryGeneratorAction.cc`
- `ActionInitialization.hh`
- `ActionInitialization.cc`

Sonra derle:

```bash
cd <geant4-build-dir>
cmake --build . --target exampleTestEm13 -j
```

Çalıştır:

```bash
./exampleTestEm13 run_cs137_gauss.mac
```

## 4) Neden daha gerçekçi?

Önceki basit modele göre şunlar eklendi:

- HPGe dedektörde yalnızca tek silindir yerine **endcap + vakum + Be pencere + dead layer + aktif kristal** ayrımı
- Kaynağın aktif noktasının geometrik olarak modellenmesi
- Enerjinin tek değer değil gaussian dağılımdan çekilmesi
- Kabin gövdesinin modele dahil edilmesi

## 5) İstersen bir sonraki adım

Bunu bir adım daha ileri götürüp:

- Cs-137 tam bozunma şeması (çoklu hat + branching ratio)
- Gerçek HPGe üretici datasheet'e göre kristal boyut/boşluk
- Sayım elektroniği için ek Gaussian broadening (FWHM(E) modeli)

ekleyebilirim.
