#import "template/template.typ": *
#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Accelerating SmolVLA on an FPGA Using Allo],
  abstract: [
    TODO: 
  ],
  authors: (
    (
      name: Sam,
      department: [Attention Layer],
      email: "srb343@cornell.edu"
    ),
    (
      name: Ezra,
      department: [Attention Layer],
      email: "er495@cornell.edu"
    ),
    (
      name: Stanley,
      department: [MLP Layer],
      email: "ss3679@cornell.edu"
    ),
    (
      name: Isabella,
      department: [MLP Layer],
      email: "isf9@cornell.edu"
    ),
  ),
  index-terms: ("HLS", "Allo"),
  bibliography: bibliography("./refs.bib"),
  figure-supplement: [Fig.],
)

#include "sections/01-introduction.typ"
#include "sections/02-background.typ"
#include "sections/03-analytical-modeling.typ"
#include "sections/04-workloads-and-hardware.typ"
#include "sections/05-implementations.typ"
#include "sections/06-evaluation.typ"
#include "sections/07-discussion.typ"
#include "sections/08-related-work.typ"
#include "sections/09-conclusion.typ"
