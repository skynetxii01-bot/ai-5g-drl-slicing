cmake_minimum_required(VERSION 3.10)

set(name slice-rl-sim)
set(source_files
  slice-rl-sim.cc
  slice-env.cc
)

build_exec(
  EXECNAME ${name}
  SOURCEFILE ${source_files}
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libinternet}
    ${libpoint-to-point}
    ${libapplications}
    ${libmobility}
    ${libnr}
    ${libopengym}
)
