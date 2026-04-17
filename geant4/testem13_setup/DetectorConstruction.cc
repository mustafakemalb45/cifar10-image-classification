#include "DetectorConstruction.hh"

#include "G4Box.hh"
#include "G4Colour.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4SubtractionSolid.hh"
#include "G4SystemOfUnits.hh"
#include "G4Tubs.hh"
#include "G4VisAttributes.hh"

G4VPhysicalVolume* DetectorConstruction::Construct() {
  auto* nist = G4NistManager::Instance();

  // ----- Malzemeler -----
  auto* air = nist->FindOrBuildMaterial("G4_AIR");
  auto* pb = nist->FindOrBuildMaterial("G4_Pb");
  auto* ge = nist->FindOrBuildMaterial("G4_Ge");
  auto* al = nist->FindOrBuildMaterial("G4_Al");
  auto* cu = nist->FindOrBuildMaterial("G4_Cu");
  auto* be = nist->FindOrBuildMaterial("G4_Be");
  auto* naI = nist->FindOrBuildMaterial("G4_SODIUM_IODIDE");
  auto* polycarb = nist->FindOrBuildMaterial("G4_POLYCARBONATE");
  auto* vacuum = nist->FindOrBuildMaterial("G4_Galactic");

  // ----- Dünya hacmi -----
  const G4double worldX = 80.0 * cm;
  const G4double worldY = 80.0 * cm;
  const G4double worldZ = 140.0 * cm;

  auto* worldSolid = new G4Box("World", worldX / 2.0, worldY / 2.0, worldZ / 2.0);
  auto* worldLV = new G4LogicalVolume(worldSolid, air, "World");
  auto* worldPV = new G4PVPlacement(nullptr, {}, worldLV, "World", nullptr, false, 0, true);

  // ----- Metal kabin: dış kutu - iç boşluk -----
  const G4double cabinOuterX = 60.0 * cm;
  const G4double cabinOuterY = 60.0 * cm;
  const G4double cabinOuterZ = 110.0 * cm;
  const G4double cabinWall = 3.0 * mm;

  auto* cabinOuter =
      new G4Box("CabinOuter", cabinOuterX / 2.0, cabinOuterY / 2.0, cabinOuterZ / 2.0);
  auto* cabinInner = new G4Box("CabinInner", (cabinOuterX - 2.0 * cabinWall) / 2.0,
                               (cabinOuterY - 2.0 * cabinWall) / 2.0,
                               (cabinOuterZ - 2.0 * cabinWall) / 2.0);
  auto* cabinShellSolid = new G4SubtractionSolid("CabinShell", cabinOuter, cabinInner);
  auto* cabinShellLV = new G4LogicalVolume(cabinShellSolid, cu, "CabinShell");
  new G4PVPlacement(nullptr, {}, cabinShellLV, "CabinShell", worldLV, false, 0, true);

  auto* cabinAirLV = new G4LogicalVolume(cabinInner, air, "CabinAir");
  new G4PVPlacement(nullptr, {}, cabinAirLV, "CabinAir", worldLV, false, 0, true);

  // Eksen: +Z yönü kaynak -> kolimatör -> numune -> dedektör
  // Kaynak merkezi referansı
  const G4double sourceCenterZ = -42.0 * cm;

  // ----- Kaynak kapsülü (polikarbonat) -----
  const G4double srcWidth = 22.0 * mm;
  const G4double srcHeight = 12.0 * mm;
  const G4double srcThickness = 3.0 * mm;
  auto* sourceCapsuleSolid =
      new G4Box("SourceCapsule", srcWidth / 2.0, srcHeight / 2.0, srcThickness / 2.0);
  auto* sourceCapsuleLV = new G4LogicalVolume(sourceCapsuleSolid, polycarb, "SourceCapsule");
  new G4PVPlacement(nullptr, {0, 0, sourceCenterZ}, sourceCapsuleLV, "SourceCapsule", cabinAirLV,
                    false, 0, true);

  // Kaynağın aktif bölgesi (merkezde küçük disk)
  const G4double activeRadius = 1.0 * mm;
  const G4double activeThickness = 0.2 * mm;
  auto* activeSpotSolid =
      new G4Tubs("SourceActiveSpot", 0.0, activeRadius, activeThickness / 2.0, 0.0, 360.0 * deg);
  auto* activeSpotLV = new G4LogicalVolume(activeSpotSolid, polycarb, "SourceActiveSpot");
  new G4PVPlacement(nullptr, {0, 0, sourceCenterZ}, activeSpotLV, "SourceActiveSpot", cabinAirLV,
                    false, 0, true);

  // ----- Kurşun kolimatör: 9x9x9 cm, 2 mm delik -----
  const G4double colSize = 9.0 * cm;
  const G4double holeRadius = 1.0 * mm;
  auto* colBox = new G4Box("CollimatorBox", colSize / 2.0, colSize / 2.0, colSize / 2.0);
  auto* colHole =
      new G4Tubs("CollimatorHole", 0.0, holeRadius, colSize / 2.0 + 1.0 * mm, 0.0, 360.0 * deg);
  auto* colSolid = new G4SubtractionSolid("CollimatorSolid", colBox, colHole);
  auto* colLV = new G4LogicalVolume(colSolid, pb, "Collimator");

  const G4double sourceFrontFaceZ = sourceCenterZ + srcThickness / 2.0;
  const G4double colCenterZ = sourceFrontFaceZ + colSize / 2.0;
  new G4PVPlacement(nullptr, {0, 0, colCenterZ}, colLV, "Collimator", cabinAirLV, false, 0, true);

  // ----- Numune bölgesi (kolimatör çıkışında) -----
  const G4double sampleThickness = 6.0 * cm;
  const G4double sampleRadius = 2.5 * cm;
  auto* sampleSolid = new G4Tubs("SampleRegion", 0.0, sampleRadius, sampleThickness / 2.0, 0.0,
                                 360.0 * deg);
  auto* sampleLV = new G4LogicalVolume(sampleSolid, air, "SampleRegion");
  const G4double sampleCenterZ = colCenterZ + colSize / 2.0 + sampleThickness / 2.0;
  new G4PVPlacement(nullptr, {0, 0, sampleCenterZ}, sampleLV, "SampleRegion", cabinAirLV, false, 0,
                    true);

  // ----- HPGe dedektör başlığı (daha gerçekçi katmanlı model) -----
  // İstenen kaynak-dedektör mesafesi 38 cm: aktif Ge kristal merkezleri arası korunuyor.
  const G4double hpgeCrystalRadius = 3.25 * cm;
  const G4double hpgeCrystalLength = 7.0 * cm;
  const G4double hpgeDeadLayer = 0.7 * mm;

  const G4double hpgeCenterZ = sourceCenterZ + 38.0 * cm;

  // Al endcap
  const G4double endcapOuterRadius = 4.0 * cm;
  const G4double endcapLength = 14.0 * cm;
  const G4double endcapWall = 1.5 * mm;
  auto* endcapOuter =
      new G4Tubs("HPGeEndcapOuter", 0.0, endcapOuterRadius, endcapLength / 2.0, 0.0, 360.0 * deg);
  auto* endcapInner = new G4Tubs("HPGeEndcapInner", 0.0, endcapOuterRadius - endcapWall,
                                 (endcapLength - 2.0 * endcapWall) / 2.0, 0.0, 360.0 * deg);
  auto* endcapSolid = new G4SubtractionSolid("HPGeEndcap", endcapOuter, endcapInner);
  auto* endcapLV = new G4LogicalVolume(endcapSolid, al, "HPGeEndcap");
  new G4PVPlacement(nullptr, {0, 0, hpgeCenterZ}, endcapLV, "HPGeEndcap", cabinAirLV, false, 0,
                    true);

  // Endcap içi vakum
  auto* endcapVacuumSolid = new G4Tubs("EndcapVacuum", 0.0, endcapOuterRadius - endcapWall,
                                       (endcapLength - 2.0 * endcapWall) / 2.0, 0.0, 360.0 * deg);
  auto* endcapVacuumLV = new G4LogicalVolume(endcapVacuumSolid, vacuum, "EndcapVacuum");
  new G4PVPlacement(nullptr, {}, endcapVacuumLV, "EndcapVacuum", endcapLV, false, 0, true);

  // Be pencere (giriş tarafı)
  const G4double beWindowThickness = 0.6 * mm;
  auto* beWindowSolid = new G4Tubs("BeWindow", 0.0, endcapOuterRadius - endcapWall,
                                   beWindowThickness / 2.0, 0.0, 360.0 * deg);
  auto* beWindowLV = new G4LogicalVolume(beWindowSolid, be, "BeWindow");
  const G4double beWindowZLocal = -endcapLength / 2.0 + endcapWall + beWindowThickness / 2.0;
  new G4PVPlacement(nullptr, {0, 0, beWindowZLocal}, beWindowLV, "BeWindow", endcapVacuumLV,
                    false, 0, true);

  // Ge kristali + dead layer
  auto* geDeadSolid = new G4Tubs("HPGeDeadLayer", 0.0, hpgeCrystalRadius,
                                 hpgeCrystalLength / 2.0, 0.0, 360.0 * deg);
  auto* geDeadLV = new G4LogicalVolume(geDeadSolid, ge, "HPGeDeadLayer");

  auto* geActiveSolid = new G4Tubs("HPGeActiveCrystal", 0.0, hpgeCrystalRadius - hpgeDeadLayer,
                                   (hpgeCrystalLength - 2.0 * hpgeDeadLayer) / 2.0, 0.0,
                                   360.0 * deg);
  auto* geActiveLV = new G4LogicalVolume(geActiveSolid, ge, "HPGeActiveCrystal");

  const G4double crystalZLocal = -1.0 * cm;  // endcap içinde öne doğru hafif ofset
  new G4PVPlacement(nullptr, {0, 0, crystalZLocal}, geDeadLV, "HPGeDeadLayer", endcapVacuumLV,
                    false, 0, true);
  new G4PVPlacement(nullptr, {}, geActiveLV, "HPGeActiveCrystal", geDeadLV, false, 0, true);

  // Alternatif NaI(Tl): aynı doğrusal eksende park hacmi
  const G4double naiRadius = 3.8 * cm;
  const G4double naiLength = 7.6 * cm;
  auto* naiSolid = new G4Tubs("NaICrystal", 0.0, naiRadius, naiLength / 2.0, 0.0, 360.0 * deg);
  auto* naiLV = new G4LogicalVolume(naiSolid, naI, "NaICrystal");
  const G4double naiCenterZ = hpgeCenterZ + 20.0 * cm;
  new G4PVPlacement(nullptr, {0, 0, naiCenterZ}, naiLV, "NaICrystal", cabinAirLV, false, 0, true);

  // ----- Görselleştirme -----
  worldLV->SetVisAttributes(G4VisAttributes::GetInvisible());

  auto* cabinVis = new G4VisAttributes(G4Colour(0.55, 0.35, 0.2));
  cabinVis->SetForceSolid(false);
  cabinShellLV->SetVisAttributes(cabinVis);

  auto* sourceVis = new G4VisAttributes(G4Colour(1.0, 1.0, 0.0));
  sourceVis->SetForceSolid(true);
  sourceCapsuleLV->SetVisAttributes(sourceVis);

  auto* activeSpotVis = new G4VisAttributes(G4Colour(1.0, 0.0, 0.0));
  activeSpotVis->SetForceSolid(true);
  activeSpotLV->SetVisAttributes(activeSpotVis);

  auto* colVis = new G4VisAttributes(G4Colour(0.35, 0.35, 0.35));
  colVis->SetForceSolid(true);
  colLV->SetVisAttributes(colVis);

  auto* sampleVis = new G4VisAttributes(G4Colour(0.2, 0.8, 1.0, 0.3));
  sampleVis->SetForceSolid(true);
  sampleLV->SetVisAttributes(sampleVis);

  auto* endcapVis = new G4VisAttributes(G4Colour(0.8, 0.8, 0.8));
  endcapVis->SetForceSolid(false);
  endcapLV->SetVisAttributes(endcapVis);

  auto* geActiveVis = new G4VisAttributes(G4Colour(0.0, 0.8, 0.0));
  geActiveVis->SetForceSolid(true);
  geActiveLV->SetVisAttributes(geActiveVis);

  auto* geDeadVis = new G4VisAttributes(G4Colour(0.0, 0.5, 0.0, 0.25));
  geDeadVis->SetForceSolid(true);
  geDeadLV->SetVisAttributes(geDeadVis);

  auto* naiVis = new G4VisAttributes(G4Colour(0.0, 0.0, 1.0));
  naiVis->SetForceSolid(true);
  naiLV->SetVisAttributes(naiVis);

  return worldPV;
}
