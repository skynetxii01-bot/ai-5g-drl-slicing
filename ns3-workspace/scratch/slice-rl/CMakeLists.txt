# CMake definition for the custom NS-3 scratch target.
build_lib(
  LIBNAME slice-rl
  SOURCE_FILES
    slice-env.cc
    slice-rl-sim.cc
  HEADER_FILES
    slice-env.h
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libinternet}
    ${libapplications}
    ${libmobility}
    ${libpoint-to-point}
    ${libflow-monitor}
    ${libnr}
    ${libopengym}
)
