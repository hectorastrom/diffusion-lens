# Instructions for Editing HTML


### Adding a new citation
Citation system managed by a small JS script at the bottom that runs on load. Thanks Gemini!

* To add a new citation in the text:

1. Just add `<a href="ref_name" class="cite"></a>.` You don't need to write [NUMBER] inside the tag.
1. Add a new `<li>` to the references list at the bottom:
```HTML
  <li id="ref_name">      Author, Title, etc.  </li>
```
  * I recommend [zoterobib](https://zbib.org/) for citation generation.