#let todo(
  who,
  body,
  done: 0%,
) = {
  set align(center)
  rect(
    fill: color.mix(
      (color.oklab(red).lighten(100%), 100% - done), 
      (color.oklab(green).lighten(100%), done)
    ),
    inset: 8pt,
    radius: 8pt,
    stroke: 1pt,
    width: 80%,
    if done >= 100% {
      align(left)[
        #text(size: 14pt)[
          *Written By: *#who #h(1fr) *(DONE)*
        ]
      ]
    } else {
      align(left)[
        #text(size: 14pt)[
          *TODO: *#who  #h(1fr) *(#done)*
        ] \ #text[
          #body
        ]
      ]
    },
  )
}
