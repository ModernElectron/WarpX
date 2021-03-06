/* Copyright 2019-2020 Axel Huebl, Junmin Gu, Maxence Thevenet
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_OPEN_PMD_H_
#define WARPX_OPEN_PMD_H_

#include "Particles/WarpXParticleContainer.H"

#include "Diagnostics/ParticleDiag/ParticleDiag_fwd.H"

#include <AMReX_AmrParticles.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuAllocators.H>
#include <AMReX_ParIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>
#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>

#ifdef WARPX_USE_OPENPMD
#   include <openPMD/openPMD.hpp>
#endif

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

//
// helper class
//
class Timer
{
public:
  Timer(const char* tag) {m_Tag = tag; m_Start = amrex::second();}
  ~Timer() {
      m_End = amrex::second();
      amrex::ParallelDescriptor::ReduceRealMax(m_End, amrex::ParallelDescriptor::IOProcessorNumber());
      amrex::Print()<<m_Tag<<" took:"<<m_End - m_Start<<" seconds\n";
  }
private:
  amrex::Real m_Start;
  amrex::Real m_End;
  std::string m_Tag;
};


//
//
class WarpXParticleCounter
{
public:
  using ParticleContainer = typename amrex::AmrParticleContainer<0, 0, PIdx::nattribs, 0, amrex::PinnedArenaAllocator>;
  using ParticleIter = typename amrex::ParIter<0, 0, PIdx::nattribs, 0, amrex::PinnedArenaAllocator>;

  WarpXParticleCounter(ParticleContainer* pc);
  unsigned long GetTotalNumParticles() {return m_Total;}

  std::vector<unsigned long long> m_ParticleOffsetAtRank;
  std::vector<unsigned long long> m_ParticleSizeAtRank;
private:
  /** get the offset in the overall particle id collection
  *
  * @param[out] numParticles particles on this processor  / amrex fab
  * @param[out] offset particle offset over all, mpi-global amrex fabs
  * @param[out] sum number of all particles from all amrex fabs
  */
  void GetParticleOffsetOfProcessor(const long& numParticles,
                    unsigned long long& offset,
                    unsigned long long& sum)  const ;


  int m_MPIRank = 0;
  int m_MPISize = 1;

  unsigned long long m_Total = 0;

  std::vector<unsigned long long> m_ParticleCounterByLevel;
};


#ifdef WARPX_USE_OPENPMD
//
//
/** Writer logic for openPMD particles and fields */
class WarpXOpenPMDPlot
{
public:
  using ParticleContainer = typename amrex::AmrParticleContainer<0, 0, PIdx::nattribs, 0, amrex::PinnedArenaAllocator>;
  using ParticleIter = typename amrex::ParConstIter<0, 0, PIdx::nattribs, 0, amrex::PinnedArenaAllocator>;

  /** Initialize openPMD I/O routines
   *
   * @param ie  iteration encoding from openPMD: "group, file, variable"
   * @param filetype file backend, e.g. "bp" or "h5"
   * @param operator_type openPMD-api backend operator (compressor) for ADIOS2
   * @param operator_parameters openPMD-api backend operator parameters for ADIOS2
   * @param fieldPMLdirections PML field solver, @see WarpX::getPMLdirections()
   */
  WarpXOpenPMDPlot (openPMD::IterationEncoding ie,
                    std::string filetype,
                    std::string operator_type,
                    std::map< std::string, std::string > operator_parameters,
                    std::vector<bool> fieldPMLdirections);

  ~WarpXOpenPMDPlot ();

  /** Set Iteration Step for the series
   *
   * @note If an iteration has been written, then it will give a warning
   *
   */
  void SetStep (int ts, const std::string& dirPrefix, int file_min_digits,
                bool isBTD=false);

  /** Close the step
   *
   * Signal that no further updates will be written for the step.
   */
  void CloseStep (bool isBTD = false, bool isLastBTDFlush = false);

  void WriteOpenPMDParticles (const amrex::Vector<ParticleDiag>& particle_diags);

  void WriteOpenPMDFieldsAll (
              const std::vector<std::string>& varnames,
              const amrex::Vector<amrex::MultiFab>& mf,
              amrex::Vector<amrex::Geometry>& geom,
              const int iteration, const double time,
              bool isBTD = false,
              const amrex::Geometry& full_BTD_snapshot=amrex::Geometry() ) const;


private:
  void Init (openPMD::Access access, bool isBTD);


  inline openPMD::Iteration& GetIteration(int iteration) const
  {
    // so BTD will be able to revisit previous steps, so we do not use steps with these two encodings,
    if (  (openPMD::IterationEncoding::fileBased == m_Encoding ) ||
          (openPMD::IterationEncoding::groupBased == m_Encoding )  )
    {
        openPMD::Iteration& it = m_Series->iterations[iteration];
        return it;
    } else {
        auto iterations = m_Series->writeIterations();
        openPMD::Iteration& it = iterations[iteration];
        return it;
    }
  }


  /** This function does initial setup for the fields when interation is newly created
   *  @param[in] meshes   The meshes in a series
   *  @param[in] full_geom The geometry
   */
  void SetupFields(  openPMD::Container< openPMD::Mesh >& meshes, amrex::Geometry& full_geom  ) const;

  void SetupMeshComp( openPMD::Mesh& mesh,
                      amrex::Geometry& full_geom,
                      openPMD::MeshRecordComponent& mesh_comp
                     ) const;

  void GetMeshCompNames( int meshLevel,
                         const std::string& varname,
                         std::string& field_name,
                         std::string& comp_name ) const;

  /** This function sets up the entries for storing the particle positions, global IDs, and constant records (charge, mass)
  *
  * @param[in] currSpecies Corresponding openPMD species
  * @param[in] np          Number of particles
  * @param[in] charge      Charge of the particles (note: fix for ions)
  * @param[in] mass        Mass of the particles
  */
  void SetupPos (
        openPMD::ParticleSpecies& currSpecies,
        const unsigned long long& np,
        amrex::ParticleReal const charge,
        amrex::ParticleReal const mass) const ;

  /** This function sets up the entries for particle properties
   *
   * @param[in] currSpecies The openPMD species
   * @param[in] write_real_comp The real attribute ids, from WarpX
   * @param[in] real_comp_names The real attribute names, from WarpX
   * @param[in] write_int_comp The int attribute ids, from WarpX
   * @param[in] int_comp_names The int attribute names, from WarpX
   * @param[in] np  Number of particles
   */
  void SetupRealProperties (openPMD::ParticleSpecies& currSpecies,
               const amrex::Vector<int>& write_real_comp,
               const amrex::Vector<std::string>& real_comp_names,
               const amrex::Vector<int>& write_int_comp,
               const amrex::Vector<std::string>& int_comp_names,
               unsigned long long np) const;

  /** This function saves the values of the entries for particle properties
   *
   * @param[in] pti WarpX particle iterator
   * @param[in] currSpecies The openPMD species to save to
   * @param[in] offset offset to start saving  the particle iterator contents
   * @param[in] write_real_comp The real attribute ids, from WarpX
   * @param[in] real_comp_names The real attribute names, from WarpX
   * @param[in] write_int_comp The int attribute ids, from WarpX
   * @param[in] int_comp_names The int attribute names, from WarpX
   */
  void SaveRealProperty (ParticleIter& pti, //int, int,
            openPMD::ParticleSpecies& currSpecies,
            unsigned long long offset,
            const amrex::Vector<int>& write_real_comp,
            const amrex::Vector<std::string>& real_comp_names,
            const amrex::Vector<int>& write_int_comp,
            const amrex::Vector<std::string>& int_comp_names) const;

  /** This function saves the plot file
   *
   * @param[in] pc WarpX particle container
   * @param[in] name species name
   * @param[in] iteration timestep
   * @param[in] write_real_comp The real attribute ids, from WarpX
   * @param[in] real_comp_names The real attribute names, from WarpX
   * @param[in] write_int_comp The int attribute ids, from WarpX
   * @param[in] int_comp_names The int attribute names, from WarpX
   * @param[in] charge         Charge of the particles (note: fix for ions)
   * @param[in] mass           Mass of the particles
   */
  void DumpToFile (ParticleContainer* pc,
            const std::string& name,
            int iteration,
            const amrex::Vector<int>& write_real_comp,
            const amrex::Vector<int>& write_int_comp,
            const amrex::Vector<std::string>& real_comp_names,
            const amrex::Vector<std::string>&  int_comp_names,
            amrex::ParticleReal const charge,
            amrex::ParticleReal const mass) const;

  /** Get the openPMD-api filename for openPMD::Series
   *
   * No need for ts in the file name, openPMD handles steps (iterations).
   *
   * @param[inout] filepath the path and filename for openPMD::Series
   *               passes a prefix path in and appends the filename
   * @return pure filename w/o path
   */
  std::string GetFileName (std::string& filepath);

  std::unique_ptr<openPMD::Series> m_Series;

  /** This is the output directory
   *
   * This usually does not yet end in a `/`.
   * It does not yet include the file prefix of the openPMD series, which will
   * be appended by the GetFileName function.
   */
  std::string m_dirPrefix;

  /** This is the minimum number of digits in the step number that is used as the
   * suffix for file names when doing file based encoding */
  int m_file_min_digits;

  int m_MPIRank = 0;
  int m_MPISize = 1;

  int m_NumSoARealAttributes = PIdx::nattribs; //! WarpX' additional real particle attributes in SoA
  int m_NumAoSRealAttributes = 0; //! WarpX definition: no additional real attributes in particle AoS

  //int m_NumSoAIntAttributes = PIdx::nattribs; //! WarpX' additional int particle attributes in SoA
  int m_NumAoSIntAttributes = 0; //! WarpX definition: no additional int attributes in particle AoS

  openPMD::IterationEncoding m_Encoding = openPMD::IterationEncoding::fileBased;
  std::string m_OpenPMDFileType = "bp"; //! MPI-parallel openPMD backend: bp or h5
  std::string m_OpenPMDoptions = "{}"; //! JSON option string for openPMD::Series constructor
  int m_CurrentStep  = -1;

  // meta data
  std::vector< bool > m_fieldPMLdirections; //! @see WarpX::getPMLdirections()
};
#endif // WARPX_USE_OPENPMD

#endif // WARPX_OPEN_PMD_H
