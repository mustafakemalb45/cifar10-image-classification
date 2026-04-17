#include "PrimaryGeneratorAction.hh"

#include "G4Event.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4RandomDirection.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "Randomize.hh"

#include <cmath>

PrimaryGeneratorAction::PrimaryGeneratorAction() {
  fParticleGun = new G4ParticleGun(1);

  auto* particleTable = G4ParticleTable::GetParticleTable();
  auto* gamma = particleTable->FindParticle("gamma");
  fParticleGun->SetParticleDefinition(gamma);

  // Kaynak aktif bölge merkezinden salım (DetectorConstruction ile aynı eksen)
  fParticleGun->SetParticlePosition(G4ThreeVector(0., 0., -42.0 * cm));

  // Kolimatör eksenine hizalı başlangıç yönü (+Z)
  fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0., 0., 1.));
}

PrimaryGeneratorAction::~PrimaryGeneratorAction() { delete fParticleGun; }

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event) {
  // Cs-137 için sabit enerji yerine gaussian dağılım:
  // Ana gamma hattı: 661.657 keV
  // Sigma: 0.80 keV (kaynak + elektronik yayılımını temsil eden pratik seçim)
  constexpr G4double meanEnergy = 661.657 * keV;
  constexpr G4double sigmaEnergy = 0.80 * keV;

  G4double sampledEnergy = G4RandGauss::shoot(meanEnergy, sigmaEnergy);

  // Negatif/çok düşük enerji örneklenirse fiziksel alt sınıra çek.
  if (sampledEnergy < 1.0 * keV) {
    sampledEnergy = 1.0 * keV;
  }

  // Kolimatör kabulünü artırmak için küçük açısal yayılım (yarım açı ~0.5 derece)
  const G4double thetaMax = 0.5 * deg;
  const G4double theta = std::acos(1.0 - G4UniformRand() * (1.0 - std::cos(thetaMax)));
  const G4double phi = 2.0 * CLHEP::pi * G4UniformRand();
  G4ThreeVector dir(std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi),
                    std::cos(theta));

  fParticleGun->SetParticleMomentumDirection(dir.unit());
  fParticleGun->SetParticleEnergy(sampledEnergy);
  fParticleGun->GeneratePrimaryVertex(event);
}
