#let move-figures-to-appendix = true
#let stored-figures = state("stored-figures", ())

#show figure: it => {
  if move-figures-to-appendix {
    stored-figures.update(figs => figs + (it,))
    none
  } else {
    it
  }
}

#set heading(numbering: "1.1.")
#set figure(supplement: it => {
  if it.func() == table {
    "Tab."
  } else {
    "Fig. (func: " + repr(it.func()) + ")"
  }
})

= Section One
First figure here.
#figure([Fake Image 1], caption: [Caption One]) <fig1>

= Section Two
Second figure here.
#figure(table[| A | B |], caption: [Caption Two]) <tab1>

= Appendix
#if move-figures-to-appendix [
  #context {
    let figs = stored-figures.get()
    for fig in figs [
      #fig
    ]
  }
]
