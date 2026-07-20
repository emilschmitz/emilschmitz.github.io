# After the site is written, emit a plain-text copy of each published
# post's source markdown at <permalink>/raw (and <permalink>/raw.md).
#
# Example: /oopsbench/raw  →  full _posts/…-oopsbench.md source

require "fileutils"

Jekyll::Hooks.register :site, :post_write do |site|
  site.posts.docs.each do |post|
    next if post.data["published"] == false

    source = File.read(post.path)
    rel_dir = post.url.to_s.sub(%r{\A/}, "").sub(%r{/\z}, "")
    next if rel_dir.empty?

    dest_dir = File.join(site.dest, rel_dir)
    FileUtils.mkdir_p(dest_dir)

    File.write(File.join(dest_dir, "raw"), source)
    File.write(File.join(dest_dir, "raw.md"), source)
  end
end
